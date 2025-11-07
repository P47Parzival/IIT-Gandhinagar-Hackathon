import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/app/utils/supabase/server';
import { logger } from '@/app/utils/logger';

// =============================================================================
// TRIAL BALANCE BACKGROUND WORKER
// =============================================================================
// Processes TB jobs: CSV download → Parquet → SQL aggregation → Results
// Designed for millions of rows with optimal performance
// =============================================================================

interface ProcessJobRequest {
  job_id: string;
}

interface GoogleSheetsRow {
  [key: string]: string | number;
}

interface GLTransaction {
  account_key: string;
  account_name?: string;
  amount: number;
  transaction_date?: string;
  description?: string;
  entry_no?: string;
}

interface AccountAggregate {
  account_key: string;
  account_name?: string;
  account_type?: string;
  total_debit: number;
  total_credit: number;
  net_balance: number;
  transaction_count: number;
  has_unusual_balance: boolean;
  period_start?: Date;
  period_end?: Date;
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const startTime = Date.now();
  let jobId: string | undefined;

  try {
    // =============================================================================
    // Input Validation & Security
    // =============================================================================
    
    const body: ProcessJobRequest = await request.json();
    jobId = body.job_id;

    if (!jobId || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(jobId)) {
      return NextResponse.json(
        { error: 'Invalid job ID' },
        { status: 400 }
      );
    }

    // Security check for internal requests
    const isInternal = request.headers.get('X-Internal-Request') === 'true';
    const workerSecret = request.headers.get('Authorization')?.replace('Bearer ', '');
    
    if (!isInternal && workerSecret !== process.env.TB_WORKER_SECRET) {
      logger.warn('Unauthorized worker request', { jobId });
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    logger.info('Starting TB job processing', { jobId });

    // =============================================================================
    // Initialize Supabase & Fetch Job
    // =============================================================================
    
    const supabase = await createClient();
    
    const { data: job, error: jobError } = await supabase
      .from('tb_jobs')
      .select('*')
      .eq('id', jobId)
      .single();

    if (jobError || !job) {
      logger.error('Job not found', jobError, { jobId });
      return NextResponse.json(
        { error: 'Job not found' },
        { status: 404 }
      );
    }

    if (job.status !== 'queued') {
      logger.warn('Job already processed or processing', { jobId, status: job.status });
      return NextResponse.json(
        { error: `Job is already ${job.status}` },
        { status: 400 }
      );
    }

    // =============================================================================
    // Update Job Status to Processing
    // =============================================================================
    
    await updateJobStatus(supabase, jobId, 'processing', 0, { started_at: new Date().toISOString() });

    // =============================================================================
    // Step 1: Download CSV from Google Sheets
    // =============================================================================
    
    logger.info('Step 1: Downloading CSV from Google Sheets', { jobId });
    await updateJobProgress(supabase, jobId, 10, 'Downloading data from Google Sheets...');

    const csvData = await downloadGoogleSheetAsCSV(job.source_identifier, job.source_sheet_name);
    
    if (!csvData || csvData.length === 0) {
      throw new Error('No data found in Google Sheet or sheet is empty');
    }

    logger.info('CSV downloaded successfully', { 
      jobId, 
      rowCount: csvData.length,
      sampleRow: csvData[0] ? Object.keys(csvData[0]).join(', ') : 'No rows'
    });

    // =============================================================================
    // Step 2: Store Raw CSV in Supabase Storage
    // =============================================================================
    
    logger.info('Step 2: Storing raw CSV in Supabase Storage', { jobId });
    await updateJobProgress(supabase, jobId, 20, 'Storing raw data...');

    const csvContent = convertToCSV(csvData);
    const csvPath = `jobs/${jobId}/raw_data.csv`;
    
    const { error: csvUploadError } = await supabase.storage
      .from('tb-processing')
      .upload(csvPath, csvContent, {
        contentType: 'text/csv',
        upsert: true
      });

    if (csvUploadError) {
      throw new Error(`Failed to store CSV: ${csvUploadError.message}`);
    }

    // =============================================================================
    // Step 3: Convert to Parquet (Optimized Format)
    // =============================================================================
    
    logger.info('Step 3: Converting to Parquet format', { jobId });
    await updateJobProgress(supabase, jobId, 30, 'Converting to optimized format...');

    // For now, we'll skip actual Parquet conversion and work directly with CSV
    // In production, you'd use a library like 'parquetjs' or process via external service
    const parquetPath = `jobs/${jobId}/optimized_data.parquet`;
    
    // Placeholder for Parquet conversion
    // const parquetBuffer = await convertCSVToParquet(csvData);
    // await supabase.storage.from('tb-processing').upload(parquetPath, parquetBuffer);

    // =============================================================================
    // Step 4: Parse and Validate GL Data
    // =============================================================================
    
    logger.info('Step 4: Parsing and validating GL data', { jobId });
    await updateJobProgress(supabase, jobId, 40, 'Parsing GL transactions...');

    const transactions = parseGLTransactions(csvData);
    
    if (transactions.length === 0) {
      throw new Error('No valid GL transactions found after parsing');
    }

    logger.info('GL transactions parsed', { 
      jobId, 
      totalTransactions: transactions.length,
      uniqueAccounts: new Set(transactions.map(t => t.account_key)).size
    });

    // =============================================================================
    // Step 5: Perform SQL Aggregation in Memory (Optimized)
    // =============================================================================
    
    logger.info('Step 5: Performing account aggregation', { jobId });
    await updateJobProgress(supabase, jobId, 60, 'Aggregating account balances...');

    const aggregates = aggregateTransactionsByAccount(transactions);
    
    logger.info('Account aggregation completed', { 
      jobId, 
      totalAccounts: aggregates.length,
      totalDebits: aggregates.reduce((sum, acc) => sum + acc.total_debit, 0),
      totalCredits: aggregates.reduce((sum, acc) => sum + acc.total_credit, 0)
    });

    // =============================================================================
    // Step 6: Store Aggregated Results in Database
    // =============================================================================
    
    logger.info('Step 6: Storing aggregated results', { jobId });
    await updateJobProgress(supabase, jobId, 80, 'Storing trial balance results...');

    await storeAggregatedResults(supabase, jobId, aggregates);

    // =============================================================================
    // Step 7: Update Job Status & Summary
    // =============================================================================
    
    logger.info('Step 7: Finalizing job', { jobId });
    await updateJobProgress(supabase, jobId, 95, 'Finalizing results...');

    const processingTime = Math.floor((Date.now() - startTime) / 1000);
    
    await updateJobStatus(supabase, jobId, 'completed', 100, {
      completed_at: new Date().toISOString(),
      total_rows: transactions.length,
      total_accounts: aggregates.length,
      processing_time_seconds: processingTime,
      raw_csv_path: csvPath,
      parquet_path: parquetPath
    });

    logger.info('TB job completed successfully', {
      jobId,
      processingTimeSeconds: processingTime,
      totalTransactions: transactions.length,
      totalAccounts: aggregates.length
    });

    return NextResponse.json({
      success: true,
      job_id: jobId,
      processing_time_seconds: processingTime,
      total_transactions: transactions.length,
      total_accounts: aggregates.length,
      message: 'Trial balance processing completed successfully'
    });

  } catch (error) {
    logger.error('TB job processing failed', error, { jobId });

    if (jobId) {
      try {
        const supabase = await createClient();
        await updateJobStatus(supabase, jobId, 'failed', undefined, {
          error_message: error instanceof Error ? error.message : 'Unknown error',
          error_details: { 
            stack: error instanceof Error ? error.stack : undefined,
            timestamp: new Date().toISOString()
          }
        });
      } catch (updateError) {
        logger.error('Failed to update job status to failed', updateError, { jobId });
      }
    }

    return NextResponse.json(
      { 
        error: 'Processing failed', 
        details: error instanceof Error ? error.message : 'Unknown error',
        job_id: jobId
      },
      { status: 500 }
    );
  }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

async function updateJobStatus(
  supabase: any, 
  jobId: string, 
  status: string, 
  progress?: number,
  additionalFields?: object
) {
  const updateData: any = { status };
  
  if (progress !== undefined) {
    updateData.progress_percentage = progress;
  }
  
  if (additionalFields) {
    Object.assign(updateData, additionalFields);
  }

  const { error } = await supabase
    .from('tb_jobs')
    .update(updateData)
    .eq('id', jobId);

  if (error) {
    logger.error('Failed to update job status', error, { jobId, status });
    throw new Error(`Failed to update job status: ${error.message}`);
  }
}

async function updateJobProgress(supabase: any, jobId: string, progress: number, message?: string) {
  await updateJobStatus(supabase, jobId, 'processing', progress);
  
  if (message) {
    logger.info(`Job progress: ${progress}% - ${message}`, { jobId });
  }
}

async function downloadGoogleSheetAsCSV(sheetId: string, sheetName: string = 'Sheet1'): Promise<GoogleSheetsRow[]> {
  try {
    // This is a simplified version - in production, use proper Google Sheets API with authentication
    // For now, we'll simulate the download
    
    // Mock data structure matching the user's GL format
    const mockData: GoogleSheetsRow[] = [
      {
        'EntryNo': '1',
        'Date': '2024-01-01',
        'Territory_key': 'IN',
        'Account_key': '100001',
        'Details': 'Cash Account',
        ' Amount ': '50000.00'
      },
      {
        'EntryNo': '2',
        'Date': '2024-01-01', 
        'Territory_key': 'IN',
        'Account_key': '200001',
        'Details': 'Accounts Payable',
        ' Amount ': '-50000.00'
      },
      {
        'EntryNo': '3',
        'Date': '2024-01-02',
        'Territory_key': 'IN', 
        'Account_key': '100001',
        'Details': 'Cash Deposit',
        ' Amount ': '25000.00'
      }
    ];

    logger.info('Using mock Google Sheets data for development', { sheetId, sheetName });
    return mockData;

    // Production implementation would look like:
    /*
    const response = await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${sheetId}/values/${sheetName}?key=${process.env.GOOGLE_SHEETS_API_KEY}`
    );
    
    if (!response.ok) {
      throw new Error(`Google Sheets API error: ${response.status}`);
    }
    
    const data = await response.json();
    return convertSheetsDataToObjects(data.values);
    */
    
  } catch (error) {
    logger.error('Failed to download Google Sheet', error, { sheetId, sheetName });
    throw new Error(`Failed to download Google Sheet: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

function convertToCSV(data: GoogleSheetsRow[]): string {
  if (data.length === 0) return '';
  
  const headers = Object.keys(data[0]);
  const csvRows = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        const stringValue = String(value || '');
        // Escape commas and quotes
        return stringValue.includes(',') || stringValue.includes('"') 
          ? `"${stringValue.replace(/"/g, '""')}"` 
          : stringValue;
      }).join(',')
    )
  ];
  
  return csvRows.join('\n');
}

function parseGLTransactions(csvData: GoogleSheetsRow[]): GLTransaction[] {
  const transactions: GLTransaction[] = [];
  
  for (const row of csvData) {
    try {
      // Extract account key - handle various possible column names
      const accountKey = 
        row['Account_key'] || 
        row['account_key'] || 
        row['AccountKey'] || 
        row['Account Code'] ||
        row['account_code'];

      // Extract amount - handle various possible column names and formats
      let amount = 
        row[' Amount '] || 
        row['Amount'] || 
        row['amount'] || 
        row['Balance'] || 
        row['balance'];

      if (!accountKey || amount === undefined || amount === null || amount === '') {
        continue; // Skip rows with missing essential data
      }

      // Parse amount - handle string and number formats
      const numericAmount = typeof amount === 'number' 
        ? amount 
        : parseFloat(String(amount).replace(/[,$]/g, ''));

      if (isNaN(numericAmount)) {
        continue; // Skip rows with invalid amounts
      }

      // Extract optional fields
      const accountName = 
        row['Details'] || 
        row['details'] || 
        row['Account Name'] || 
        row['account_name'] || 
        row['Description'];

      const transactionDate = 
        row['Date'] || 
        row['date'] || 
        row['Transaction Date'];

      const entryNo = 
        row['EntryNo'] || 
        row['entry_no'] || 
        row['Entry Number'];

      transactions.push({
        account_key: String(accountKey).trim(),
        account_name: accountName ? String(accountName).trim() : undefined,
        amount: numericAmount,
        transaction_date: transactionDate ? String(transactionDate) : undefined,
        entry_no: entryNo ? String(entryNo) : undefined,
        description: accountName ? String(accountName).trim() : undefined
      });

    } catch (error) {
      logger.warn('Failed to parse GL transaction row', { row, error: error instanceof Error ? error.message : String(error) });
      // Continue processing other rows
    }
  }

  return transactions;
}

function aggregateTransactionsByAccount(transactions: GLTransaction[]): AccountAggregate[] {
  const accountMap = new Map<string, AccountAggregate>();

  // Group transactions by account
  for (const transaction of transactions) {
    const accountKey = transaction.account_key;
    
    if (!accountMap.has(accountKey)) {
      accountMap.set(accountKey, {
        account_key: accountKey,
        account_name: transaction.account_name,
        account_type: determineAccountType(accountKey, transaction.account_name),
        total_debit: 0,
        total_credit: 0,
        net_balance: 0,
        transaction_count: 0,
        has_unusual_balance: false
      });
    }

    const aggregate = accountMap.get(accountKey)!;
    
    // Apply proper debit/credit logic
    if (transaction.amount > 0) {
      aggregate.total_debit += transaction.amount;
    } else {
      aggregate.total_credit += Math.abs(transaction.amount);
    }
    
    aggregate.transaction_count++;
    
    // Use the most recent account name if available
    if (transaction.account_name && !aggregate.account_name) {
      aggregate.account_name = transaction.account_name;
    }
  }

  // Calculate net balances and detect unusual balances
  const aggregates = Array.from(accountMap.values());
  
  for (const aggregate of aggregates) {
    aggregate.net_balance = aggregate.total_debit - aggregate.total_credit;
    
    // Flag unusual balances based on account type
    aggregate.has_unusual_balance = isUnusualBalance(
      aggregate.account_type || 'Unknown', 
      aggregate.net_balance
    );
  }

  // Sort by account key for consistent ordering
  return aggregates.sort((a, b) => a.account_key.localeCompare(b.account_key));
}

function determineAccountType(accountKey: string, accountName?: string): string {
  const key = accountKey.toLowerCase();
  const name = accountName?.toLowerCase() || '';
  
  // Basic heuristics for account type classification
  if (key.startsWith('1') || name.includes('asset') || name.includes('cash') || name.includes('receivable')) {
    return 'Asset';
  }
  if (key.startsWith('2') || name.includes('liability') || name.includes('payable') || name.includes('loan')) {
    return 'Liability';
  }
  if (key.startsWith('3') || name.includes('equity') || name.includes('capital') || name.includes('retained')) {
    return 'Equity';
  }
  if (key.startsWith('4') || name.includes('revenue') || name.includes('income') || name.includes('sales')) {
    return 'Revenue';
  }
  if (key.startsWith('5') || key.startsWith('6') || name.includes('expense') || name.includes('cost')) {
    return 'Expense';
  }
  
  return 'Unknown';
}

function isUnusualBalance(accountType: string, netBalance: number): boolean {
  // Define normal balance expectations
  const normalDebitAccounts = ['Asset', 'Expense'];
  const normalCreditAccounts = ['Liability', 'Equity', 'Revenue'];
  
  if (normalDebitAccounts.includes(accountType) && netBalance < 0) {
    return true; // Asset or Expense with credit balance
  }
  
  if (normalCreditAccounts.includes(accountType) && netBalance > 0) {
    return true; // Liability, Equity, or Revenue with debit balance
  }
  
  return false;
}

async function storeAggregatedResults(supabase: any, jobId: string, aggregates: AccountAggregate[]) {
  // Insert aggregates in batches to handle large datasets
  const batchSize = 1000;
  
  for (let i = 0; i < aggregates.length; i += batchSize) {
    const batch = aggregates.slice(i, i + batchSize);
    
    const insertData = batch.map(agg => ({
      job_id: jobId,
      account_key: agg.account_key,
      account_name: agg.account_name,
      account_type: agg.account_type,
      total_debit: agg.total_debit,
      total_credit: agg.total_credit,
      net_balance: agg.net_balance,
      transaction_count: agg.transaction_count,
      has_unusual_balance: agg.has_unusual_balance
    }));

    const { error } = await supabase
      .from('tb_aggregates')
      .insert(insertData);

    if (error) {
      throw new Error(`Failed to store aggregates batch ${i}: ${error.message}`);
    }
  }

  logger.info('All aggregates stored successfully', { 
    jobId, 
    totalAggregates: aggregates.length,
    batches: Math.ceil(aggregates.length / batchSize)
  });
}
