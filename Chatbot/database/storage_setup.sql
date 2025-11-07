-- =============================================================================
-- SUPABASE STORAGE SETUP FOR TRIAL BALANCE PROCESSING
-- =============================================================================
-- Creates storage bucket and policies for TB file storage
-- =============================================================================

-- Create storage bucket for TB processing files
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'tb-processing',
  'tb-processing', 
  false, -- Private bucket
  104857600, -- 100MB file size limit
  ARRAY['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/octet-stream']
) ON CONFLICT (id) DO NOTHING;

-- Allow authenticated users to upload their own TB files
CREATE POLICY "Users can upload their own TB files" ON storage.objects
FOR INSERT WITH CHECK (
  bucket_id = 'tb-processing' 
  AND auth.uid()::text = (storage.foldername(name))[1]
);

-- Allow authenticated users to view their own TB files
CREATE POLICY "Users can view their own TB files" ON storage.objects
FOR SELECT USING (
  bucket_id = 'tb-processing' 
  AND auth.uid()::text = (storage.foldername(name))[1]
);

-- Allow system to manage all TB files (for background processing)
CREATE POLICY "System can manage TB files" ON storage.objects
FOR ALL USING (bucket_id = 'tb-processing');

-- Function to cleanup old TB files
CREATE OR REPLACE FUNCTION cleanup_old_tb_files()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER := 0;
    file_record RECORD;
BEGIN
    -- Delete files from jobs older than 7 days
    FOR file_record IN 
        SELECT o.name 
        FROM storage.objects o
        JOIN tb_jobs j ON o.name LIKE 'jobs/' || j.id::text || '/%'
        WHERE o.bucket_id = 'tb-processing'
        AND j.expires_at < NOW()
        AND j.status IN ('completed', 'failed')
    LOOP
        DELETE FROM storage.objects 
        WHERE bucket_id = 'tb-processing' AND name = file_record.name;
        deleted_count := deleted_count + 1;
    END LOOP;
    
    RETURN deleted_count;
END;
$$;

COMMENT ON FUNCTION cleanup_old_tb_files() IS 'Cleanup function for old TB processing files - run via cron';
