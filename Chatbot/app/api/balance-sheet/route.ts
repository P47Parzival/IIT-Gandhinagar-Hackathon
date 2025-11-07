import { NextRequest, NextResponse } from "next/server";
import { createClient } from '@/app/utils/supabase/server';
import { logger } from '@/app/utils/logger';

/**
 * Balance Sheet Assurance Integration API
 * Connects Open Rube chat interface with Balance Sheet Assurance backend
 */

const BALANCE_SHEET_API_BASE = process.env.BALANCE_SHEET_API_BASE || 'http://localhost:8000/api/v1';

export async function POST(request: NextRequest) {
  try {
    const { query, context } = await request.json();
    
    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' }, 
        { status: 400 }
      );
    }

    // Get authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json(
        { error: 'Unauthorized' }, 
        { status: 401 }
      );
    }

    logger.info('Processing balance sheet query', { 
      userId: user.id, 
      query: query.substring(0, 100) 
    });

    // Forward query to Balance Sheet Assurance API with fallback
    let balanceSheetResponse;
    try {
      const response = await fetch(`${BALANCE_SHEET_API_BASE}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          context: {
            ...context,
            user_id: user.id,
            user_email: user.email
          }
        }),
        // Add timeout to prevent hanging
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });

      if (!response.ok) {
        throw new Error(`Balance Sheet API error: ${response.status}`);
      }

      balanceSheetResponse = await response.json();
    } catch (error) {
      // Fallback when backend is unavailable
      logger.warn('Balance Sheet API unavailable, using fallback response', { 
        error: error.message 
      });
      
      balanceSheetResponse = {
        response: `I understand you're asking about balance sheet analysis: "${query}". 
        
The Balance Sheet Assurance service is currently unavailable. Here's what I can tell you based on your query:

• For GL variance analysis, check accounts with >20% variance month-over-month
• Supporting documents should be uploaded for all material GL accounts  
• Trial balance should sum to zero for validation
• Review workflows follow: Owner → Reviewer → Controller approval chain

Please start the Balance Sheet API service (port 8000) or use regular chat for general assistance.`,
        data: {
          mockKPIs: {
            hygieneScore: 85.2,
            pendingDocuments: 23,
            majorVariances: 5,
            approvalRate: 92.1
          },
          mockVariances: [
            { account: "Cash & Cash Equivalents", variance: "25.0%", status: "Under Review", amount: 150000 },
            { account: "Accounts Payable", variance: "41.7%", status: "Pending Approval", amount: -85000 },
            { account: "Revenue", variance: "25.0%", status: "Approved", amount: -250000 }
          ]
        },
        chart_data: {
          variance_trend: [
            { month: "Oct", variance: 18.5 },
            { month: "Nov", variance: 32.1 }
          ]
        },
        suggested_actions: [
          "Start Balance Sheet API service",
          "Check Supabase connection",
          "Review GL account mappings",
          "Validate trial balance totals"
        ]
      };
    }
    
    // Transform response for Open Rube format
    const transformedResponse = {
      response: balanceSheetResponse.response,
      type: 'balance_sheet_query',
      data: balanceSheetResponse.data,
      chart_data: balanceSheetResponse.chart_data,
      suggested_actions: balanceSheetResponse.suggested_actions,
      timestamp: new Date().toISOString()
    };

    logger.info('Balance sheet query processed successfully', { 
      userId: user.id,
      responseLength: balanceSheetResponse.response?.length || 0
    });

    return NextResponse.json(transformedResponse);

  } catch (error) {
    logger.error('Balance sheet query failed', error);
    return NextResponse.json(
      { error: 'Failed to process balance sheet query' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    // Get query suggestions for balance sheet queries
    let suggestions;
    try {
      const response = await fetch(`${BALANCE_SHEET_API_BASE}/chat/suggestions`, {
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      
      if (!response.ok) {
        throw new Error(`Balance Sheet API error: ${response.status}`);
      }

      suggestions = await response.json();
    } catch (error) {
      // Fallback suggestions when API is unavailable
      logger.warn('Balance Sheet API unavailable, using fallback suggestions');
      suggestions = {
        suggestions: [
          "Show GL accounts with variance > 30%",
          "Generate hygiene score report", 
          "Check supporting document status",
          "Run compliance validation",
          "Analyze variance trends"
        ]
      };
    }
    
    return NextResponse.json({
      suggestions: suggestions.suggestions,
      category: 'balance_sheet_assurance'
    });

  } catch (error) {
    logger.error('Failed to get balance sheet suggestions', error);
    return NextResponse.json(
      { error: 'Failed to get suggestions' },
      { status: 500 }
    );
  }
}
