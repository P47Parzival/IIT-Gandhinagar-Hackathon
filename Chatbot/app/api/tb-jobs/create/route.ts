import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/app/utils/supabase/server';
import { logger } from '@/app/utils/logger';

// =============================================================================
// TRIAL BALANCE JOB CREATION API
// =============================================================================
// Creates asynchronous TB processing jobs for large GL datasets
// =============================================================================

interface CreateTBJobRequest {
  source_type: 'google_sheets' | 'csv_upload' | 'excel_upload';
  source_identifier: string; // Google Sheet ID, file path, etc.
  source_sheet_name?: string;
  conversation_id?: string;
  metadata?: {
    sheet_url?: string;
    file_name?: string;
    expected_rows?: number;
    gl_period?: string;
  };
}

interface CreateTBJobResponse {
  success: boolean;
  job_id?: string;
  estimated_processing_time?: string;
  status_check_url?: string;
  error?: string;
}

export async function POST(request: NextRequest): Promise<NextResponse<CreateTBJobResponse>> {
  try {
    // =============================================================================
    // Authentication & Input Validation
    // =============================================================================
    
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      logger.warn('Unauthorized TB job creation attempt', { authError });
      return NextResponse.json(
        { success: false, error: 'Authentication required' },
        { status: 401 }
      );
    }

    const body: CreateTBJobRequest = await request.json();
    
    // Validate required fields
    if (!body.source_type || !body.source_identifier) {
      return NextResponse.json(
        { 
          success: false, 
          error: 'Missing required fields: source_type and source_identifier' 
        },
        { status: 400 }
      );
    }

    // Validate source_type
    const validSourceTypes = ['google_sheets', 'csv_upload', 'excel_upload'];
    if (!validSourceTypes.includes(body.source_type)) {
      return NextResponse.json(
        { 
          success: false, 
          error: `Invalid source_type. Must be one of: ${validSourceTypes.join(', ')}` 
        },
        { status: 400 }
      );
    }

    logger.info('Creating TB processing job', {
      userId: user.id,
      sourceType: body.source_type,
      sourceId: body.source_identifier?.substring(0, 20) + '...',
      conversationId: body.conversation_id
    });

    // =============================================================================
    // Check for Existing Active Jobs
    // =============================================================================
    
    // Prevent duplicate jobs for the same source
    const { data: existingJobs, error: checkError } = await supabase
      .from('tb_jobs')
      .select('id, status, created_at')
      .eq('user_id', user.id)
      .eq('source_identifier', body.source_identifier)
      .in('status', ['queued', 'processing'])
      .order('created_at', { ascending: false })
      .limit(1);

    if (checkError) {
      logger.error('Error checking existing jobs', checkError);
      return NextResponse.json(
        { success: false, error: 'Database error checking existing jobs' },
        { status: 500 }
      );
    }

    if (existingJobs && existingJobs.length > 0) {
      const existingJob = existingJobs[0];
      logger.info('Found existing active job', { existingJobId: existingJob.id });
      
      return NextResponse.json({
        success: true,
        job_id: existingJob.id,
        estimated_processing_time: 'Already processing',
        status_check_url: `/api/tb-jobs/${existingJob.id}`,
      });
    }

    // =============================================================================
    // Estimate Processing Time (based on source type and expected size)
    // =============================================================================
    
    const estimateProcessingTime = (sourceType: string, expectedRows?: number): string => {
      const rows = expectedRows || 10000; // Default assumption
      
      if (rows < 1000) return '30 seconds';
      if (rows < 10000) return '1-2 minutes';
      if (rows < 100000) return '2-5 minutes';
      if (rows < 1000000) return '5-15 minutes';
      return '15-30 minutes';
    };

    const estimatedTime = estimateProcessingTime(
      body.source_type, 
      body.metadata?.expected_rows
    );

    // =============================================================================
    // Create Job Record
    // =============================================================================
    
    const jobData = {
      user_id: user.id,
      conversation_id: body.conversation_id,
      source_type: body.source_type,
      source_identifier: body.source_identifier,
      source_sheet_name: body.source_sheet_name || 'Sheet1',
      status: 'queued',
      progress_percentage: 0,
      total_rows: body.metadata?.expected_rows || 0,
      created_at: new Date().toISOString(),
      expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days from now
    };

    const { data: jobResult, error: createError } = await supabase
      .from('tb_jobs')
      .insert([jobData])
      .select('id')
      .single();

    if (createError || !jobResult) {
      logger.error('Error creating TB job', createError);
      return NextResponse.json(
        { success: false, error: 'Failed to create processing job' },
        { status: 500 }
      );
    }

    const jobId = jobResult.id;

    logger.info('TB job created successfully', {
      jobId,
      userId: user.id,
      estimatedTime,
      sourceType: body.source_type
    });

    // =============================================================================
    // Trigger Background Processing (via Edge Function or Worker)
    // =============================================================================
    
    try {
      // Option 1: Call background worker via HTTP (if deployed as separate service)
      if (process.env.TB_WORKER_URL) {
        const workerResponse = await fetch(`${process.env.TB_WORKER_URL}/process`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.TB_WORKER_SECRET}`,
          },
          body: JSON.stringify({ job_id: jobId }),
        });

        if (!workerResponse.ok) {
          logger.warn('Failed to trigger background worker', {
            jobId,
            status: workerResponse.status
          });
        }
      } else {
        // Option 2: Call internal API route for processing (fallback)
        const baseUrl = request.nextUrl.origin;
        const processResponse = await fetch(`${baseUrl}/api/tb-jobs/process`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Internal-Request': 'true',
          },
          body: JSON.stringify({ job_id: jobId }),
        });

        if (!processResponse.ok) {
          logger.warn('Failed to trigger internal background processing', {
            jobId,
            status: processResponse.status
          });
        }
      }
    } catch (triggerError) {
      logger.error('Error triggering background processing', triggerError, { jobId });
      // Don't fail the job creation - worker can pick it up later
    }

    // =============================================================================
    // Return Success Response
    // =============================================================================
    
    const response: CreateTBJobResponse = {
      success: true,
      job_id: jobId,
      estimated_processing_time: estimatedTime,
      status_check_url: `/api/tb-jobs/${jobId}`,
    };

    return NextResponse.json(response, { 
      status: 201,
      headers: {
        'X-Job-ID': jobId,
        'X-Estimated-Time': estimatedTime,
      }
    });

  } catch (error) {
    logger.error('Unexpected error in TB job creation', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// =============================================================================
// GET: List User's Jobs (Optional endpoint for job management)
// =============================================================================

export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Get query parameters
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status');
    const limit = parseInt(searchParams.get('limit') || '10');
    const offset = parseInt(searchParams.get('offset') || '0');

    // Build query
    let query = supabase
      .from('tb_jobs')
      .select(`
        id,
        source_type,
        source_identifier,
        source_sheet_name,
        status,
        progress_percentage,
        total_rows,
        total_accounts,
        processing_time_seconds,
        error_message,
        created_at,
        started_at,
        completed_at
      `)
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (status) {
      query = query.eq('status', status);
    }

    const { data: jobs, error: fetchError } = await query;

    if (fetchError) {
      logger.error('Error fetching user jobs', fetchError);
      return NextResponse.json(
        { error: 'Failed to fetch jobs' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      jobs: jobs || [],
      pagination: {
        limit,
        offset,
        total: jobs?.length || 0
      }
    });

  } catch (error) {
    logger.error('Error in GET tb-jobs', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
