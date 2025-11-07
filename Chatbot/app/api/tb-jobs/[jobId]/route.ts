import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/app/utils/supabase/server';
import { logger } from '@/app/utils/logger';

// =============================================================================
// TRIAL BALANCE JOB STATUS & RESULTS API
// =============================================================================
// Fetch job status, progress, and results by jobId
// =============================================================================

interface TBJobStatusResponse {
  success: boolean;
  job?: {
    id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    progress_percentage: number;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    estimated_completion?: string;
    source_type: string;
    source_identifier: string;
    source_sheet_name: string;
    total_rows: number;
    total_accounts: number;
    processing_time_seconds: number;
    error_message?: string;
    error_details?: any;
  };
  summary?: {
    total_debits: number;
    total_credits: number;
    balance_difference: number;
    is_balanced: boolean;
    total_accounts: number;
    total_transactions: number;
    data_quality_score: number;
    period_start?: string;
    period_end?: string;
  };
  aggregates?: Array<{
    account_key: string;
    account_name?: string;
    account_type?: string;
    total_debit: number;
    total_credit: number;
    net_balance: number;
    transaction_count: number;
    has_unusual_balance: boolean;
    variance_from_previous?: number;
  }>;
  download_links?: {
    csv?: string;
    parquet?: string;
    excel?: string;
  };
  error?: string;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
): Promise<NextResponse<TBJobStatusResponse>> {
  try {
    const { jobId } = params;

    // =============================================================================
    // Authentication & Validation
    // =============================================================================
    
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      logger.warn('Unauthorized TB job status request', { jobId, authError });
      return NextResponse.json(
        { success: false, error: 'Authentication required' },
        { status: 401 }
      );
    }

    if (!jobId || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(jobId)) {
      return NextResponse.json(
        { success: false, error: 'Invalid job ID format' },
        { status: 400 }
      );
    }

    logger.debug('Fetching TB job status', { jobId, userId: user.id });

    // =============================================================================
    // Fetch Job Details with Optimized Query
    // =============================================================================
    
    // Use the custom function for complete job data in a single query
    const { data: completeJobData, error: fetchError } = await supabase
      .rpc('get_tb_job_with_results', { p_job_id: jobId });

    if (fetchError) {
      logger.error('Error fetching job data', fetchError, { jobId });
      return NextResponse.json(
        { success: false, error: 'Database error fetching job' },
        { status: 500 }
      );
    }

    if (!completeJobData || !completeJobData.job) {
      logger.warn('Job not found', { jobId, userId: user.id });
      return NextResponse.json(
        { success: false, error: 'Job not found' },
        { status: 404 }
      );
    }

    const jobData = completeJobData.job;
    const summaryData = completeJobData.summary;
    const aggregatesData = completeJobData.aggregates || [];

    // Verify user owns this job
    if (jobData.user_id !== user.id) {
      logger.warn('Unauthorized access to job', { jobId, userId: user.id, jobUserId: jobData.user_id });
      return NextResponse.json(
        { success: false, error: 'Access denied' },
        { status: 403 }
      );
    }

    // =============================================================================
    // Calculate Estimated Completion Time (for queued/processing jobs)
    // =============================================================================
    
    const calculateEstimatedCompletion = (job: any): string | undefined => {
      if (job.status === 'completed' || job.status === 'failed') {
        return undefined;
      }

      const now = new Date();
      const created = new Date(job.created_at);
      const elapsed = Math.floor((now.getTime() - created.getTime()) / 1000); // seconds

      if (job.status === 'queued') {
        // Estimate based on queue position and typical processing time
        const estimatedSeconds = Math.max(60, job.total_rows / 1000 * 10); // ~10ms per row
        return new Date(now.getTime() + estimatedSeconds * 1000).toISOString();
      }

      if (job.status === 'processing' && job.progress_percentage > 0) {
        // Estimate based on current progress
        const remainingPercent = 100 - job.progress_percentage;
        const avgTimePerPercent = elapsed / job.progress_percentage;
        const estimatedRemainingSeconds = remainingPercent * avgTimePerPercent;
        return new Date(now.getTime() + estimatedRemainingSeconds * 1000).toISOString();
      }

      return undefined;
    };

    // =============================================================================
    // Generate Download Links (for completed jobs)
    // =============================================================================
    
    const generateDownloadLinks = async (job: any) => {
      if (job.status !== 'completed') return undefined;

      const links: any = {};

      try {
        // Generate signed URLs for file downloads
        if (job.raw_csv_path) {
          const { data: csvUrl } = await supabase.storage
            .from('tb-processing')
            .createSignedUrl(job.raw_csv_path, 3600); // 1 hour expiry

          if (csvUrl?.signedUrl) {
            links.csv = csvUrl.signedUrl;
          }
        }

        if (job.parquet_path) {
          const { data: parquetUrl } = await supabase.storage
            .from('tb-processing')
            .createSignedUrl(job.parquet_path, 3600);

          if (parquetUrl?.signedUrl) {
            links.parquet = parquetUrl.signedUrl;
          }
        }

        // Generate Excel export link (dynamic generation)
        links.excel = `/api/tb-jobs/${jobId}/export/excel`;

      } catch (error) {
        logger.warn('Error generating download links', { jobId, error: error instanceof Error ? error.message : String(error) });
      }

      return Object.keys(links).length > 0 ? links : undefined;
    };

    const downloadLinks = await generateDownloadLinks(jobData);

    // =============================================================================
    // Format Response
    // =============================================================================
    
    const response: TBJobStatusResponse = {
      success: true,
      job: {
        id: jobData.id,
        status: jobData.status,
        progress_percentage: jobData.progress_percentage || 0,
        created_at: jobData.created_at,
        started_at: jobData.started_at,
        completed_at: jobData.completed_at,
        estimated_completion: calculateEstimatedCompletion(jobData),
        source_type: jobData.source_type,
        source_identifier: jobData.source_identifier,
        source_sheet_name: jobData.source_sheet_name,
        total_rows: jobData.total_rows || 0,
        total_accounts: jobData.total_accounts || 0,
        processing_time_seconds: jobData.processing_time_seconds || 0,
        error_message: jobData.error_message,
        error_details: jobData.error_details,
      },
    };

    // Add summary data if available
    if (summaryData) {
      response.summary = {
        total_debits: parseFloat(summaryData.total_debits || '0'),
        total_credits: parseFloat(summaryData.total_credits || '0'),
        balance_difference: parseFloat(summaryData.balance_difference || '0'),
        is_balanced: summaryData.is_balanced || false,
        total_accounts: summaryData.total_accounts || 0,
        total_transactions: summaryData.total_transactions || 0,
        data_quality_score: summaryData.data_quality_score || 0,
        period_start: summaryData.period_start,
        period_end: summaryData.period_end,
      };
    }

    // Add aggregates data if available and not too large
    if (aggregatesData && aggregatesData.length > 0) {
      // Limit aggregates in response to prevent huge payloads
      const maxAggregates = 1000;
      response.aggregates = aggregatesData.slice(0, maxAggregates).map((agg: any) => ({
        account_key: agg.account_key,
        account_name: agg.account_name,
        account_type: agg.account_type,
        total_debit: parseFloat(agg.total_debit || '0'),
        total_credit: parseFloat(agg.total_credit || '0'),
        net_balance: parseFloat(agg.net_balance || '0'),
        transaction_count: agg.transaction_count || 0,
        has_unusual_balance: agg.has_unusual_balance || false,
        variance_from_previous: agg.variance_from_previous ? parseFloat(agg.variance_from_previous) : undefined,
      }));

      // If we had to truncate, add a note
      if (aggregatesData.length > maxAggregates) {
        logger.info('Truncated aggregates in response', { 
          jobId, 
          totalAggregates: aggregatesData.length, 
          returnedAggregates: maxAggregates 
        });
      }
    }

    // Add download links if available
    if (downloadLinks) {
      response.download_links = downloadLinks;
    }

    logger.debug('TB job status retrieved successfully', {
      jobId,
      status: jobData.status,
      totalAccounts: response.summary?.total_accounts || 0,
      isBalanced: response.summary?.is_balanced
    });

    return NextResponse.json(response, {
      headers: {
        'Cache-Control': jobData.status === 'completed' || jobData.status === 'failed' 
          ? 'public, max-age=3600' // Cache completed jobs for 1 hour
          : 'no-cache', // Don't cache active jobs
        'X-Job-Status': jobData.status,
        'X-Progress': String(jobData.progress_percentage || 0),
      }
    });

  } catch (error) {
    logger.error('Unexpected error in TB job status API', error, { jobId: params.jobId });
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// =============================================================================
// DELETE: Cancel/Delete Job
// =============================================================================

export async function DELETE(
  request: NextRequest,
  { params }: { params: { jobId: string } }
): Promise<NextResponse> {
  try {
    const { jobId } = params;
    
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      return NextResponse.json(
        { success: false, error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Only allow deletion of queued jobs or failed jobs
    const { data: job, error: fetchError } = await supabase
      .from('tb_jobs')
      .select('id, status, user_id')
      .eq('id', jobId)
      .eq('user_id', user.id)
      .single();

    if (fetchError || !job) {
      return NextResponse.json(
        { success: false, error: 'Job not found' },
        { status: 404 }
      );
    }

    if (job.status === 'processing') {
      return NextResponse.json(
        { success: false, error: 'Cannot delete job that is currently processing' },
        { status: 400 }
      );
    }

    // Delete the job (cascades to aggregates and summary)
    const { error: deleteError } = await supabase
      .from('tb_jobs')
      .delete()
      .eq('id', jobId)
      .eq('user_id', user.id);

    if (deleteError) {
      logger.error('Error deleting job', deleteError, { jobId });
      return NextResponse.json(
        { success: false, error: 'Failed to delete job' },
        { status: 500 }
      );
    }

    logger.info('TB job deleted', { jobId, userId: user.id });

    return NextResponse.json({ 
      success: true, 
      message: 'Job deleted successfully' 
    });

  } catch (error) {
    logger.error('Error in DELETE tb-job', error, { jobId: params.jobId });
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}
