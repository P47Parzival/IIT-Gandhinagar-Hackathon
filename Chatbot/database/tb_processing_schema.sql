-- =============================================================================
-- TRIAL BALANCE PROCESSING SYSTEM - DATABASE SCHEMA
-- =============================================================================
-- This schema supports asynchronous, scalable TB processing for millions of rows
-- =============================================================================

-- Drop existing tables if they exist (for development only)
-- DROP TABLE IF EXISTS tb_aggregates;
-- DROP TABLE IF EXISTS tb_jobs;

-- =============================================================================
-- TABLE: tb_jobs - Job Queue and Status Tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS tb_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Job Identification
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    conversation_id VARCHAR(255),
    
    -- Source Data Info
    source_type VARCHAR(50) NOT NULL DEFAULT 'google_sheets', -- 'google_sheets', 'csv_upload', 'excel_upload'
    source_identifier TEXT NOT NULL, -- Google Sheet ID, file path, etc.
    source_sheet_name VARCHAR(255) DEFAULT 'Sheet1',
    
    -- Processing Status
    status VARCHAR(50) NOT NULL DEFAULT 'queued', -- 'queued', 'processing', 'completed', 'failed'
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    
    -- File Storage
    raw_csv_path TEXT, -- Path in Supabase Storage
    parquet_path TEXT, -- Path in Supabase Storage for optimized format
    
    -- Processing Metadata
    total_rows INTEGER DEFAULT 0,
    total_accounts INTEGER DEFAULT 0,
    processing_time_seconds INTEGER DEFAULT 0,
    
    -- Error Handling
    error_message TEXT,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days'), -- Auto-cleanup after 7 days
    
    -- Indexing
    CONSTRAINT valid_status CHECK (status IN ('queued', 'processing', 'completed', 'failed'))
);

-- =============================================================================
-- TABLE: tb_aggregates - Final Trial Balance Results
-- =============================================================================
CREATE TABLE IF NOT EXISTS tb_aggregates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to Job
    job_id UUID NOT NULL REFERENCES tb_jobs(id) ON DELETE CASCADE,
    
    -- Account Information
    account_key VARCHAR(255) NOT NULL,
    account_name VARCHAR(500),
    account_type VARCHAR(100), -- 'Asset', 'Liability', 'Equity', 'Revenue', 'Expense'
    
    -- Financial Data
    total_debit DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    total_credit DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    net_balance DECIMAL(18,2) NOT NULL DEFAULT 0.00, -- (total_debit - total_credit)
    
    -- Metadata
    transaction_count INTEGER DEFAULT 0,
    period_start DATE,
    period_end DATE,
    
    -- Validation Flags
    has_unusual_balance BOOLEAN DEFAULT FALSE, -- e.g., negative cash
    variance_from_previous DECIMAL(10,4), -- Percentage change from last period
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_account_per_job UNIQUE(job_id, account_key)
);

-- =============================================================================
-- TABLE: tb_summary - Job-Level Summary Statistics
-- =============================================================================
CREATE TABLE IF NOT EXISTS tb_summary (
    job_id UUID PRIMARY KEY REFERENCES tb_jobs(id) ON DELETE CASCADE,
    
    -- Trial Balance Validation
    total_debits DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    total_credits DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    balance_difference DECIMAL(18,2) NOT NULL DEFAULT 0.00, -- Should be 0.00 for balanced TB
    is_balanced BOOLEAN GENERATED ALWAYS AS (ABS(balance_difference) < 0.01) STORED,
    
    -- Statistics
    total_accounts INTEGER DEFAULT 0,
    total_transactions INTEGER DEFAULT 0,
    largest_account_balance DECIMAL(18,2) DEFAULT 0.00,
    smallest_account_balance DECIMAL(18,2) DEFAULT 0.00,
    accounts_with_unusual_balances INTEGER DEFAULT 0,
    
    -- Period Information
    period_start DATE,
    period_end DATE,
    
    -- Quality Score (0-100)
    data_quality_score INTEGER DEFAULT 0 CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- INDEXES for Performance
-- =============================================================================

-- tb_jobs indexes
CREATE INDEX IF NOT EXISTS idx_tb_jobs_user_id ON tb_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_tb_jobs_status ON tb_jobs(status);
CREATE INDEX IF NOT EXISTS idx_tb_jobs_created_at ON tb_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tb_jobs_expires_at ON tb_jobs(expires_at) WHERE status IN ('completed', 'failed');

-- tb_aggregates indexes
CREATE INDEX IF NOT EXISTS idx_tb_aggregates_job_id ON tb_aggregates(job_id);
CREATE INDEX IF NOT EXISTS idx_tb_aggregates_account_key ON tb_aggregates(account_key);
CREATE INDEX IF NOT EXISTS idx_tb_aggregates_net_balance ON tb_aggregates(net_balance DESC);

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable RLS
ALTER TABLE tb_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE tb_aggregates ENABLE ROW LEVEL SECURITY;
ALTER TABLE tb_summary ENABLE ROW LEVEL SECURITY;

-- tb_jobs policies
CREATE POLICY "Users can view own TB jobs" ON tb_jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own TB jobs" ON tb_jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own TB jobs" ON tb_jobs
    FOR UPDATE USING (auth.uid() = user_id);

-- tb_aggregates policies
CREATE POLICY "Users can view own TB aggregates" ON tb_aggregates
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM tb_jobs 
            WHERE tb_jobs.id = tb_aggregates.job_id 
            AND tb_jobs.user_id = auth.uid()
        )
    );

CREATE POLICY "System can insert TB aggregates" ON tb_aggregates
    FOR INSERT WITH CHECK (TRUE); -- Background worker needs unrestricted insert

-- tb_summary policies
CREATE POLICY "Users can view own TB summary" ON tb_summary
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM tb_jobs 
            WHERE tb_jobs.id = tb_summary.job_id 
            AND tb_jobs.user_id = auth.uid()
        )
    );

CREATE POLICY "System can manage TB summary" ON tb_summary
    FOR ALL WITH CHECK (TRUE); -- Background worker needs full access

-- =============================================================================
-- FUNCTIONS for Common Operations
-- =============================================================================

-- Function to get job with aggregates
CREATE OR REPLACE FUNCTION get_tb_job_with_results(p_job_id UUID)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'job', row_to_json(j.*),
        'summary', row_to_json(s.*),
        'aggregates', COALESCE(
            (SELECT json_agg(row_to_json(a.*)) FROM tb_aggregates a WHERE a.job_id = p_job_id),
            '[]'::json
        )
    )
    INTO result
    FROM tb_jobs j
    LEFT JOIN tb_summary s ON s.job_id = j.id
    WHERE j.id = p_job_id;
    
    RETURN result;
END;
$$;

-- Function to cleanup expired jobs
CREATE OR REPLACE FUNCTION cleanup_expired_tb_jobs()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM tb_jobs 
    WHERE expires_at < NOW() 
    AND status IN ('completed', 'failed');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- =============================================================================
-- TRIGGERS for Automation
-- =============================================================================

-- Update tb_summary when tb_aggregates change
CREATE OR REPLACE FUNCTION update_tb_summary()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO tb_summary (
        job_id,
        total_debits,
        total_credits,
        balance_difference,
        total_accounts,
        largest_account_balance,
        smallest_account_balance,
        accounts_with_unusual_balances
    )
    SELECT 
        NEW.job_id,
        COALESCE(SUM(total_debit), 0),
        COALESCE(SUM(total_credit), 0),
        COALESCE(SUM(total_debit), 0) - COALESCE(SUM(total_credit), 0),
        COUNT(*),
        COALESCE(MAX(ABS(net_balance)), 0),
        COALESCE(MIN(ABS(net_balance)), 0),
        COUNT(*) FILTER (WHERE has_unusual_balance = TRUE)
    FROM tb_aggregates
    WHERE job_id = NEW.job_id
    ON CONFLICT (job_id) DO UPDATE SET
        total_debits = EXCLUDED.total_debits,
        total_credits = EXCLUDED.total_credits,
        balance_difference = EXCLUDED.balance_difference,
        total_accounts = EXCLUDED.total_accounts,
        largest_account_balance = EXCLUDED.largest_account_balance,
        smallest_account_balance = EXCLUDED.smallest_account_balance,
        accounts_with_unusual_balances = EXCLUDED.accounts_with_unusual_balances,
        updated_at = NOW();
    
    RETURN NEW;
END;
$$;

CREATE TRIGGER trigger_update_tb_summary
    AFTER INSERT OR UPDATE OR DELETE ON tb_aggregates
    FOR EACH ROW
    EXECUTE FUNCTION update_tb_summary();

-- =============================================================================
-- COMMENTS for Documentation
-- =============================================================================

COMMENT ON TABLE tb_jobs IS 'Job queue for asynchronous trial balance processing';
COMMENT ON TABLE tb_aggregates IS 'Final aggregated trial balance results per account';
COMMENT ON TABLE tb_summary IS 'Job-level summary statistics and validation results';
COMMENT ON FUNCTION get_tb_job_with_results(UUID) IS 'Returns complete job data with results in single query';
COMMENT ON FUNCTION cleanup_expired_tb_jobs() IS 'Cleanup function for expired jobs - run via cron';
