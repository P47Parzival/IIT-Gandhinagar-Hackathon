import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from 'crypto';
import { streamText, stepCountIs } from 'ai';
import { google } from '@ai-sdk/google';
import { experimental_createMCPClient as createMCPClient } from 'ai';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { createClient } from '@/app/utils/supabase/server';
import {
  createConversation,
  addMessage,
  generateConversationTitle
} from '@/app/utils/chat-history';
import { getComposio } from "@/app/utils/composio";
import { logger } from '@/app/utils/logger';

type MCPTools = Awaited<ReturnType<Awaited<ReturnType<typeof createMCPClient>>['tools']>>;

interface MCPSessionCache {
  session: { url: string; sessionId: string };
  client: Awaited<ReturnType<typeof createMCPClient>>;
  tools: MCPTools;
}

// Session cache to store MCP sessions per chat session per user
const sessionCache = new Map<string, MCPSessionCache>();

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { messages, conversationId, internalUserId, internalUserEmail } = body;

    const internalToken = request.headers.get('x-internal-agent-token');
    const internalTokenSecret = process.env.CHATBOT_INTERNAL_TOKEN || '';
    const isInternalRequest = !!internalToken && internalToken === internalTokenSecret;
    
    if (!messages) {
      return NextResponse.json(
        { error: 'messages is required' }, 
        { status: 400 }
      );
    }

    // Get authenticated user from server-side session
    const supabase = await createClient();
    let user: { id: string; email?: string | null } | null = null;

    if (!isInternalRequest) {
      const { data, error } = await supabase.auth.getUser();
      user = data?.user ?? null;

      if (error || !user) {
        return NextResponse.json(
          { error: 'Unauthorized - Please sign in' }, 
          { status: 401 }
        );
      }
    } else {
      user = {
        id: internalUserId || 'internal-django-bridge',
        email: internalUserEmail || null
      };
      logger.info('Processing internal agent request from bridge', {
        internalUserId: user.id,
        internalUserEmail,
      });
    }

    let userEmail = (user?.email || internalUserEmail || '').trim();
    if (!userEmail) {
      if (!isInternalRequest) {
        return NextResponse.json(
          { error: 'User email not found' }, 
          { status: 400 }
        );
      }
      userEmail = 'internal@bridge.local';
    }

    if (!user) {
      return NextResponse.json(
        { error: 'User context unavailable' },
        { status: 401 }
      );
    }

    logger.info('User authenticated', { userId: user.id });

    let currentConversationId: string = conversationId ?? '';
    const latestMessage = messages[messages.length - 1];
    const shouldPersistMessages = !isInternalRequest;
    const isFirstMessage = !currentConversationId;

    // Create new conversation if this is the first message
    if (isFirstMessage) {
      if (shouldPersistMessages) {
        const title = generateConversationTitle(latestMessage.content);
        const createdConversationId = await createConversation(user.id, title);
        
        if (!createdConversationId) {
          return NextResponse.json(
            { error: 'Failed to create conversation' }, 
            { status: 500 }
          );
        }
        currentConversationId = createdConversationId;
      } else {
        currentConversationId = randomUUID();
      }
    }

    if (!currentConversationId) {
      currentConversationId = randomUUID();
    }

    // Save user message to database
    if (shouldPersistMessages && currentConversationId) {
      await addMessage(
        currentConversationId,
        user.id,
        latestMessage.content,
        'user'
      );
    }

    logger.info('Starting Tool Router Agent execution', { conversationId: currentConversationId });

    // Create a unique session key based on user and conversation
    const sessionKey = `${user.id}-${currentConversationId}`;
    
    // We'll create the Tool Router session after checking connections
    // This ensures we can pass the correct connected account IDs
    let composio, mcpSession, mcpClient, tools;
    const COMPOSIO_USER_ID = 'pg-test-604fcca1-2eb4-4314-9907-75826a1622cd';

    // Detect if this is a financial/accounting query
    const isAccountingQuery = messages.some((msg: { content: string }) => 
      /general ledger|trial balance|GL|TB|debit|credit|accounting|financial|balance sheet|variance|audit|compliance/i.test(msg.content)
    );

    // Detect if this is specifically a Trial Balance generation request
    const lastMessage = messages[messages.length - 1]?.content || '';
    const isTBGenerationRequest = /convert.*general ledger.*trial|generate.*trial balance|create.*trial balance|GL.*to.*TB|general ledger.*into.*trial/i.test(lastMessage);

    // Check if user is requesting TB generation from Google Sheets
    const googleSheetsTBRequest = isTBGenerationRequest && /google sheet|sheet|spreadsheet/i.test(lastMessage);
    
    // Extract Google Sheet reference if present
    const googleSheetMatch = lastMessage.match(/sheet.*named?\s*['"]*([^'"\s,]+)['"]*|google sheet\s*['"]*([^'"\s,]+)['"]*|spreadsheet.*['"]*([^'"\s,]+)['"]*/i);
    const sheetName = googleSheetMatch ? (googleSheetMatch[1] || googleSheetMatch[2] || googleSheetMatch[3]) : null;

    // =============================================================================
    // HANDLE TRIAL BALANCE GENERATION REQUESTS VIA JOB SYSTEM
    // =============================================================================
    
    if (googleSheetsTBRequest && sheetName) {
      logger.info('Detected TB generation request - creating job', {
        userId: user.id,
        conversationId: currentConversationId,
        sheetName
      });

      try {
        // Create TB processing job
        const jobResponse = await fetch(`${request.nextUrl.origin}/api/tb-jobs/create`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Cookie': request.headers.get('cookie') || '', // Forward auth cookies
          },
          body: JSON.stringify({
            source_type: 'google_sheets',
            source_identifier: sheetName, // This would be the actual Sheet ID in production
            source_sheet_name: 'Sheet1',
            conversation_id: currentConversationId,
            metadata: {
              sheet_url: `https://docs.google.com/spreadsheets/d/${sheetName}`,
              file_name: `${sheetName}_GL.csv`,
              gl_period: new Date().toISOString().substring(0, 7) // Current month
            }
          })
        });

        if (!jobResponse.ok) {
          throw new Error(`Failed to create TB job: ${jobResponse.status}`);
        }

        const jobData = await jobResponse.json();

        if (jobData.success && jobData.job_id) {
          // Save user message to database
          if (shouldPersistMessages && currentConversationId) {
            if (shouldPersistMessages && currentConversationId) {
              await addMessage(currentConversationId, user.id, lastMessage, 'user');
            }
          }

          // Create and return immediate response with job details
          const jobStatusUrl = `${request.nextUrl.origin}/api/tb-jobs/${jobData.job_id}`;
          
          const immediateResponse = `🚀 **Trial Balance Processing Started!**

I've created a **background job** to process your General Ledger from Google Sheet "${sheetName}" into a Trial Balance report. This approach ensures fast, accurate processing even for millions of rows!

## 📊 **Job Details**
- **Job ID**: \`${jobData.job_id}\`
- **Estimated Time**: ${jobData.estimated_processing_time}
- **Status**: Processing in background

## 🔍 **What's Happening**
1. ✅ **Job Created** - Your request is queued
2. 🔄 **Data Download** - Fetching GL from Google Sheets  
3. 📊 **Account Aggregation** - Grouping by account with CPA-level precision
4. ✅ **Trial Balance Generation** - Creating balanced report
5. 📈 **Results Ready** - Download links and analysis

## 📱 **Check Status**
You can check progress anytime by asking:
> *"What's the status of job ${jobData.job_id}?"*

Or visit: [Job Status](${jobStatusUrl})

## ⚡ **Meanwhile...**
Feel free to ask me other questions! I can help with:
- 📧 **Email automation** with Composio
- 📱 **Slack integration** 
- 📄 **Document creation** in Google Docs
- 🔗 **Any of the 500+ connected apps**

I'll notify you as soon as your Trial Balance is ready! 🎯

---
*This new system processes large GLs efficiently while keeping our chat fast for everything else.*`;

          // Save assistant response to database
          if (shouldPersistMessages && currentConversationId) {
            await addMessage(currentConversationId, user.id, immediateResponse, 'assistant');
          }

          // Return the immediate response
          return new Response(immediateResponse, {
            headers: {
              'Content-Type': 'text/plain',
              'X-Conversation-Id': currentConversationId,
              'X-TB-Job-ID': jobData.job_id,
              'X-Job-Status-URL': jobStatusUrl,
            },
          });
        }
      } catch (jobError) {
        logger.error('Failed to create TB job - falling back to regular chat', jobError);
        // Continue with regular chat processing as fallback
      }
    }

    // =============================================================================
    // HANDLE ZOHO CRM QUERIES VIA CUSTOM OAUTH
    // =============================================================================
    
    // Detect if this is a Zoho CRM query
    const isZohoCRMQuery = /zoho\s*crm|zoho\s*leads|zoho\s*contacts|zoho\s*deals|zoho\s*accounts|zoho\s*campaigns|zoho\s*opportunities/i.test(lastMessage);
    
    logger.info('Zoho CRM query detection', {
      lastMessage: lastMessage,
      isZohoCRMQuery: isZohoCRMQuery,
      detectionPattern: 'zoho\\s*crm|zoho\\s*leads|zoho\\s*contacts|zoho\\s*deals|zoho\\s*accounts|zoho\\s*campaigns|zoho\\s*opportunities'
    });
    
    // Helper function to create Tool Router session
    async function createToolRouterSession(isZohoCRMSpecific = false, zohoConnectionId?: string) {
      if (sessionCache.has(sessionKey)) {
        logger.info('Clearing cached session to recreate with correct settings', { sessionKey });
        sessionCache.delete(sessionKey);
      }
      
      logger.info('Creating Tool Router session', { 
        sessionKey, 
        composioUserId: COMPOSIO_USER_ID,
        isZohoCRMSpecific,
        zohoConnectionId
      });
      composio = getComposio();

      let sessionConfig: any = {};
      
      if (isZohoCRMSpecific) {
        const ZOHO_AUTH_CONFIG_ID = 'ac_q9q5L3lpE9YS';
        // For Zoho queries, specify only Zoho toolkit and force the known connection to be used
        sessionConfig = {
          toolkits: [
            {
              toolkit: 'zoho',
              authConfigId: ZOHO_AUTH_CONFIG_ID
            }
          ],
          manuallyManageConnections: true
        };
        logger.info('Creating Tool Router session with Zoho toolkit for CRM query', {
          composioUserId: COMPOSIO_USER_ID,
          zohoConnectionId,
          config: sessionConfig,
          note: zohoConnectionId
            ? 'Binding session to existing Zoho auth config to prevent new connection flow'
            : 'Zoho connection id not provided - session will rely on auth config'
        });
      } else {
        // For non-Zoho queries, allow all toolkits
        sessionConfig = {
          toolkits: []
        };
        logger.info('Creating Tool Router session with all toolkits for general query', {
          composioUserId: COMPOSIO_USER_ID
        });
      }

      // Create Tool Router session with explicit connection information
      mcpSession = await composio.experimental.toolRouter.createSession(COMPOSIO_USER_ID, sessionConfig);

      const url = new URL(mcpSession.url);
      logger.debug('Tool Router session created', { 
        sessionId: mcpSession.sessionId, 
        url: url.toString()
      });

      mcpClient = await createMCPClient({
        transport: new StreamableHTTPClientTransport(url, {
          sessionId: mcpSession.sessionId,
        }),
      });

      tools = await mcpClient.tools();
      
      // Cache the session, client, and tools for this chat
      sessionCache.set(sessionKey, { session: mcpSession, client: mcpClient, tools });
    }

    if (isZohoCRMQuery) {
      logger.info('Detected Zoho CRM query - checking connection status', {
        userId: user.id,
        conversationId: currentConversationId,
        query: lastMessage.substring(0, 100)
      });

      try {
        // Check if user has Zoho CRM connection using correct Composio User ID
        const COMPOSIO_USER_ID = 'pg-test-604fcca1-2eb4-4314-9907-75826a1622cd';
        const composio = getComposio();
        
        logger.info('Fetching connected accounts for Composio User ID', { 
          composioUserId: COMPOSIO_USER_ID 
        });
        
        const connectedAccounts = await composio.connectedAccounts.list({
          userIds: [COMPOSIO_USER_ID]
        });

        logger.info('Connected accounts result', {
          totalAccounts: connectedAccounts.items?.length || 0,
          accounts: connectedAccounts.items?.map(acc => ({
            id: acc.id,
            toolkit: acc.toolkit?.slug,
            status: acc.status
          })) || []
        });

        // Check if Zoho CRM is connected
        const zohoConnection = connectedAccounts.items?.find(account => 
          account.toolkit?.slug === 'zoho' && account.status === 'ACTIVE'
        );

        logger.info('Zoho connection check result', {
          zohoFound: !!zohoConnection,
          zohoConnection: zohoConnection ? {
            id: zohoConnection.id,
            toolkit: zohoConnection.toolkit?.slug,
            status: zohoConnection.status
          } : null
        });

        if (!zohoConnection) {
          logger.info('No Zoho CRM connection found - initiating OAuth flow', { userId: user.id });

          // Save user message first
          if (shouldPersistMessages && currentConversationId) {
            await addMessage(currentConversationId, user.id, lastMessage, 'user');
          }

          try {
            // Use the existing Zoho CRM auth config ID - NEVER CREATE NEW ONES
            const ZOHO_AUTH_CONFIG_ID = 'ac_q9q5L3lpE9YS';
            
            logger.info('Using existing Zoho CRM auth config for OAuth initiation', { 
              authConfigId: ZOHO_AUTH_CONFIG_ID,
              composioUserId: COMPOSIO_USER_ID,
              userEmail,
              note: 'Using hardcoded existing config - NO AUTH CONFIG CREATION'
            });

            // Initiate connection using the existing auth config from Composio dashboard
            const connReq = await composio.connectedAccounts.initiate(COMPOSIO_USER_ID, ZOHO_AUTH_CONFIG_ID, {
              allowMultiple: true
            });
            
            logger.info('Zoho CRM OAuth initiated', { 
              userId: user.id, 
              redirectUrl: connReq.redirectUrl,
              connectionId: connReq.id 
            });

            const oauthResponse = `🎯 **Zoho CRM Connection Required**

I detected you want to work with Zoho CRM, but I need to connect to your Zoho account first!

## 🔗 **Quick Setup**
1. **Click here to authorize**: [Connect Zoho CRM](${connReq.redirectUrl})
2. **Sign in** to your Zoho account
3. **Grant permissions** for CRM access
4. **Return here** and ask your question again

## 🛡️ **Secure Process**
- OAuth2 secure authentication
- Read-only access to your CRM data
- No passwords stored
- Revoke anytime from Zoho settings

## 📱 **What I can help with after connection:**
- "Show me leads from Zoho CRM"
- "Update contact status in Zoho"
- "Generate sales report from Zoho data"
- "Find deals in Zoho CRM pipeline"

Once connected, I'll be able to access your Zoho CRM data and help you with all your CRM tasks! 🚀`;

            // Save assistant response
            if (shouldPersistMessages && currentConversationId) {
              await addMessage(currentConversationId, user.id, oauthResponse, 'assistant');
            }

            return new Response(oauthResponse, {
              headers: {
                'Content-Type': 'text/plain',
                'X-Conversation-Id': currentConversationId,
                'X-Zoho-Auth-Required': 'true',
                'X-Zoho-Redirect-URL': connReq.redirectUrl || '',
              },
            });

          } catch (oauthError) {
            logger.error('Failed to initiate Zoho CRM OAuth', oauthError);
            
            const errorResponse = `❌ **Zoho CRM Setup Issue**

I detected your Zoho CRM query, but there's a configuration issue:

**Error**: ${oauthError instanceof Error ? oauthError.message : 'Unknown OAuth error'}

## 🔧 **Setup Required**
Please ensure your administrator has configured:
- ZOHO_CLIENT_ID environment variable
- ZOHO_CLIENT_SECRET environment variable  
- Zoho API application in [Zoho API Console](https://api-console.zoho.com/)

## 💡 **Alternative**
You can also connect Zoho CRM manually via the **Link Data** tab in the navigation menu.

Once configured, I'll be able to help you with all your Zoho CRM tasks! 🎯`;

            if (shouldPersistMessages && currentConversationId) {
              await addMessage(currentConversationId, user.id, errorResponse, 'assistant');
            }

            return new Response(errorResponse, {
              headers: {
                'Content-Type': 'text/plain',
                'X-Conversation-Id': currentConversationId,
                'X-Zoho-Setup-Required': 'true',
              },
            });
          }
        } else if (zohoConnection.status === 'ACTIVE') {
          logger.info('ACTIVE Zoho CRM connection found - using existing connection', {
            userId: user.id,
            connectionId: zohoConnection.id,
            toolkit: zohoConnection.toolkit?.slug,
            status: zohoConnection.status,
            note: 'NO NEW CONNECTIONS SHOULD BE CREATED'
          });
          
          // Save user message first since we're continuing with Tool Router
          if (shouldPersistMessages && currentConversationId) {
            await addMessage(currentConversationId, user.id, lastMessage, 'user');
          }
          
          logger.info('Zoho connection ACTIVE - proceeding to Tool Router with existing connection', {
            connectionId: zohoConnection.id,
            conversationId: currentConversationId,
            warning: 'Tool Router MUST use existing connection only'
          });
          
          // Connection exists and is ACTIVE, continue with Tool Router processing
          // Pass the connection ID explicitly to prevent new connection creation
        } else {
          logger.warn('Zoho connection found but not ACTIVE', {
            connectionId: zohoConnection.id,
            status: zohoConnection.status,
            note: 'Connection exists but may need reauthorization'
          });
          
          // Save user message first
          if (shouldPersistMessages && currentConversationId) {
            await addMessage(currentConversationId, user.id, lastMessage, 'user');
          }
          
          const inactiveConnectionResponse = `⚠️ **Zoho CRM Connection Found (Inactive)**

I found your Zoho CRM connection (${zohoConnection.id}), but it's in ${zohoConnection.status} status.

## 🔄 **Reauthorization Required**
Your connection needs to be refreshed. Please use the link below to reauthorize:

[Reconnect Zoho CRM](${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}?tab=linkdata)

## 💡 **Alternative**
You can also try: *"Connect me to Zoho CRM"* to force a fresh connection.

Once reauthorized, I'll be able to help with all your Zoho CRM queries! 🎯`;

          if (shouldPersistMessages && currentConversationId) {
            await addMessage(currentConversationId, user.id, inactiveConnectionResponse, 'assistant');
          }

          return new Response(inactiveConnectionResponse, {
            headers: {
              'Content-Type': 'text/plain',
              'X-Conversation-Id': currentConversationId,
              'X-Zoho-Reauth-Required': 'true',
            },
          });
        }

        // Validate connection is truly accessible before creating Tool Router session
        try {
          logger.info('Validating Zoho connection accessibility', {
            connectionId: zohoConnection.id,
            composioUserId: COMPOSIO_USER_ID
          });
          
          // Try to get connection details to ensure it's accessible
          const connectionDetails = await composio.connectedAccounts.get(zohoConnection.id);
          
          logger.info('Zoho connection validation successful', {
            connectionId: connectionDetails.id,
            toolkit: connectionDetails.toolkit?.slug,
            status: connectionDetails.status,
            authConfigId: connectionDetails.authConfig?.id,
            note: 'Connection is accessible and ACTIVE'
          });
          
        } catch (validationError) {
          logger.error('Zoho connection validation failed', validationError);
          
          const validationErrorResponse = `❌ **Zoho CRM Connection Validation Failed**

I found your Zoho CRM connection (${zohoConnection.id}), but it's not accessible for tool execution.

**Error**: ${validationError instanceof Error ? validationError.message : 'Connection validation failed'}

## 🔄 **Recommended Actions:**
1. **Reconnect**: Use the Link Data tab to reconnect Zoho CRM
2. **Clear cache**: Try refreshing the page
3. **New query**: Ask "Connect me to Zoho CRM" for fresh setup

I'll be ready once the connection is properly validated! 🎯`;

          if (shouldPersistMessages && currentConversationId) {
            await addMessage(currentConversationId, user.id, validationErrorResponse, 'assistant');
          }

          return new Response(validationErrorResponse, {
            headers: {
              'Content-Type': 'text/plain',
              'X-Conversation-Id': currentConversationId,
              'X-Zoho-Validation-Failed': 'true',
            },
          });
        }

        // Create Tool Router session specifically for Zoho CRM with existing connection
        await createToolRouterSession(true, zohoConnection.id);

      } catch (connectionCheckError) {
        logger.error('Error checking Zoho CRM connection', connectionCheckError);
        
        // Save user message first
        if (shouldPersistMessages && currentConversationId) {
          await addMessage(currentConversationId, user.id, lastMessage, 'user');
        }
        
        const connectionErrorResponse = `⚠️ **Zoho CRM Connection Check Failed**

I detected your Zoho CRM query, but I'm having trouble checking your connection status.

**Error**: ${connectionCheckError instanceof Error ? connectionCheckError.message : 'Unknown connection error'}

## 🔄 **What to try:**
1. **Wait a moment** and try your question again
2. **Check connection** via the **Link Data** tab
3. **Manual connection**: Use "Link Data" → "Add Connection" → "Zoho CRM"

## 💡 **Alternative Query**
You can also try: *"Connect me to Zoho CRM"* to force a fresh connection attempt.

I'll be ready to help once the connection is established! 🎯`;

        if (shouldPersistMessages && currentConversationId) {
          await addMessage(currentConversationId, user.id, connectionErrorResponse, 'assistant');
        }

        return new Response(connectionErrorResponse, {
          headers: {
            'Content-Type': 'text/plain',
            'X-Conversation-Id': currentConversationId,
            'X-Zoho-Connection-Error': 'true',
          },
        });
      }
    } else {
      // For non-Zoho queries, create Tool Router session with all toolkits
      logger.info('Non-Zoho query - creating standard Tool Router session');
      await createToolRouterSession(false);
    }

    // Check if user is asking for TB job status
    const jobStatusRequest = /status.*job|job.*status|check.*progress/i.test(lastMessage);
    const jobIdMatch = lastMessage.match(/job\s+([a-f0-9-]{36})|([a-f0-9-]{36})/i);
    
    if (jobStatusRequest && jobIdMatch) {
      const extractedJobId = jobIdMatch[1] || jobIdMatch[2];
      
      try {
        const statusResponse = await fetch(`${request.nextUrl.origin}/api/tb-jobs/${extractedJobId}`, {
          headers: {
            'Cookie': request.headers.get('cookie') || '',
          },
        });

        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          
          if (statusData.success) {
            // Save user message
            await addMessage(currentConversationId, user.id, lastMessage, 'user');

            // Format status response
            const job = statusData.job;
            const summary = statusData.summary;
            
            let statusMessage = `📊 **Trial Balance Job Status**\n\n`;
            statusMessage += `**Job ID**: \`${job.id}\`\n`;
            statusMessage += `**Status**: ${job.status.toUpperCase()}\n`;
            statusMessage += `**Progress**: ${job.progress_percentage}%\n`;
            statusMessage += `**Created**: ${new Date(job.created_at).toLocaleString()}\n`;

            if (job.status === 'processing') {
              statusMessage += `**Estimated Completion**: ${job.estimated_completion ? new Date(job.estimated_completion).toLocaleString() : 'Calculating...'}\n`;
            }

            if (job.status === 'completed' && summary) {
              statusMessage += `\n## ✅ **Results Summary**\n`;
              statusMessage += `- **Total Accounts**: ${summary.total_accounts.toLocaleString()}\n`;
              statusMessage += `- **Total Debits**: $${summary.total_debits.toLocaleString()}\n`;
              statusMessage += `- **Total Credits**: $${summary.total_credits.toLocaleString()}\n`;
              statusMessage += `- **Balance Status**: ${summary.is_balanced ? '✅ BALANCED' : '⚠️ IMBALANCED'}\n`;
              statusMessage += `- **Quality Score**: ${summary.data_quality_score}/100\n`;
              
              if (statusData.download_links) {
                statusMessage += `\n## 📥 **Download Results**\n`;
                if (statusData.download_links.excel) {
                  statusMessage += `- [📊 Excel Report](${statusData.download_links.excel})\n`;
                }
                if (statusData.download_links.csv) {
                  statusMessage += `- [📄 CSV Data](${statusData.download_links.csv})\n`;
                }
              }
            }

            if (job.status === 'failed') {
              statusMessage += `\n## ❌ **Error Details**\n`;
              statusMessage += `**Error**: ${job.error_message}\n`;
              statusMessage += `\nYou can try creating a new job or contact support if the issue persists.`;
            }

            // Save assistant response
            if (shouldPersistMessages && currentConversationId) {
              await addMessage(currentConversationId, user.id, statusMessage, 'assistant');
            }

            return new Response(statusMessage, {
              headers: {
                'Content-Type': 'text/plain',
                'X-Conversation-Id': currentConversationId,
              },
            });
          }
        }
      } catch (statusError) {
        logger.error('Failed to fetch job status - falling back to regular chat', statusError);
        // Continue with regular chat processing as fallback
      }
    }

    // =============================================================================
    // CONTINUE WITH REGULAR CHAT PROCESSING (All Other Capabilities)
    // =============================================================================

    const result = await streamText({
      model: google('gemini-2.5-pro'),
      tools,
      // Set precise temperature for financial calculations
      temperature: isAccountingQuery ? 0.1 : 0.3,
      system: `You are Nirva, an AI assistant with CPA-level accounting expertise that can interact with 500+ applications through Composio's Tool Router.

${isAccountingQuery ? `
🧮 **ACCOUNTING & FINANCIAL EXPERTISE MODE ACTIVATED**

You are now operating with specialized accounting knowledge. Follow these CRITICAL rules:

**FUNDAMENTAL ACCOUNTING PRINCIPLES:**
- **The Accounting Equation**: Assets = Liabilities + Equity (ALWAYS verify this holds)
- **Debit/Credit Rules**: 
  • Assets & Expenses: Debit increases, Credit decreases
  • Liabilities, Equity & Revenue: Credit increases, Debit decreases
- **Trial Balance Rule**: Total Debits MUST equal Total Credits (sum = 0)

**GENERAL LEDGER TO TRIAL BALANCE CONVERSION PROTOCOL:**
1. **NEVER ASSUME DATA STRUCTURE** - Always examine and confirm:
   - Column names and their meaning
   - Date ranges and accounting periods
   - Currency and number formats
   - Account coding system used

2. **VALIDATION REQUIREMENTS:**
   - Verify GL data completeness before processing
   - Check for duplicate entries or missing transactions
   - Ensure account codes match standard chart of accounts
   - Validate that all entries have both debit and credit components

3. **CALCULATION METHODOLOGY:**
   - Group transactions by account code/name
   - Sum debits and credits separately for each account
   - Calculate net balance (debit - credit) for each account
   - MANDATORY: Verify total debits = total credits
   - Flag any accounts with unusual balances (e.g., negative cash)

4. **ERROR PREVENTION:**
   - Ask clarifying questions about ambiguous data
   - Confirm account classifications before processing
   - Provide step-by-step calculation transparency
   - Always show the trial balance equation verification

**WHEN PROCESSING FINANCIAL DATA:**
- State your understanding of the data structure BEFORE processing
- Show sample calculations for verification
- Highlight any accounts that don't follow normal balance patterns
- Provide detailed reconciliation between source GL and final TB

**⚡ TRIAL BALANCE GENERATION SYSTEM:**
- For large GL datasets (Google Sheets conversion), I use a **background job system**
- This ensures accurate processing of millions of rows without timeouts
- Jobs provide real-time progress updates and downloadable results
- All other accounting queries are processed immediately with enhanced precision
` : ''}

            **Response Formatting Guidelines:**
            - Always format your responses using Markdown syntax
            - Use **bold** for emphasis and important points
            - Use bullet points and numbered lists for clarity
            - Format links as [text](url) so they are clickable
            - Use code blocks with \`\`\` for code snippets
            - Use inline code with \` for commands, file names, and technical terms
            - Use headings (##, ###) to organize longer responses
            - Make your responses clear, concise, and well-structured

            **Tool Execution Guidelines:**
            - Explain what you're doing before using tools
            - Provide clear feedback about the results
            - Include relevant links when appropriate
            ${isAccountingQuery ? '- For financial data: ALWAYS validate calculations and provide audit trails' : ''}
            
          `,
      messages: messages,
      stopWhen: stepCountIs(50),
      onStepFinish: (step) => {
        logger.debug('AI step completed');
        
        // Enhanced accounting-specific validation
        if (isAccountingQuery) {
          const stepText = step.text || '';
          const toolCalls = step.toolCalls || [];
          
          // Check for accounting calculation errors
          const hasAccountingCalculation = /debit|credit|balance|total|sum/i.test(stepText);
          const hasGoogleSheetsActivity = toolCalls.some(call => 
            call.toolName?.toLowerCase().includes('googlesheets') || 
            call.toolName?.toLowerCase().includes('workbench')
          );
          
          if (hasAccountingCalculation || hasGoogleSheetsActivity) {
            // Flag potential accounting errors
            const accountingErrorPatterns = [
              /error|failed|mismatch/i,
              /cannot find|not found/i,
              /invalid|incorrect/i,
              /zero sum|imbalance/i
            ];
            
            const hasError = accountingErrorPatterns.some(pattern => pattern.test(stepText));
            
            // Flag missing validation indicators
            const missingValidation = hasAccountingCalculation && !/verify|check|validate|total.*=.*total/i.test(stepText);
            
            // Flag assumption-based processing
            const hasAssumptions = /assume|assuming|probably|likely/i.test(stepText) && hasAccountingCalculation;
            
            if (hasError) {
              logger.error('Accounting calculation error detected', {
                conversationId: currentConversationId,
                errorIndicators: stepText.substring(0, 200)
              });
            }
            
            if (missingValidation) {
              logger.warn('Accounting calculation missing validation', {
                conversationId: currentConversationId,
                calculation: stepText.substring(0, 200)
              });
            }
            
            if (hasAssumptions) {
              logger.warn('Accounting process using assumptions', {
                conversationId: currentConversationId,
                assumptions: stepText.substring(0, 200)
              });
            }
            
            // Log successful validation indicators
            const hasValidation = /total debits.*=.*total credits|balance.*verified|equation.*holds/i.test(stepText);
            if (hasValidation) {
              logger.info('Accounting validation successful', {
                conversationId: currentConversationId,
                validation: stepText.substring(0, 200)
              });
            }
          }
        }
      },
      onFinish: async (event) => {
        // Save assistant response to database when streaming finishes
        if (!shouldPersistMessages || !currentConversationId) {
          return;
        }

        try {
          const result = await addMessage(
            currentConversationId,
            user.id,
            event.text,
            'assistant'
          );

          if (!result) {
            logger.warn('Failed to save assistant message to database', {
              conversationId: currentConversationId,
              userId: user.id,
              textLength: event.text.length
            });
          } else {
            logger.debug('Assistant message saved to database', {
              conversationId: currentConversationId,
              messageLength: event.text.length
            });
          }
        } catch (error) {
          logger.error('Error saving assistant message', error, {
            conversationId: currentConversationId,
            userId: user.id
          });
        }
      },
    });

    // Return streaming response
    return result.toTextStreamResponse({
      headers: {
        'X-Conversation-Id': currentConversationId,
      },
    });
  } catch (error) {
    logger.error('Error in chat endpoint', error);
    return NextResponse.json(
      { error: 'Failed to process chat request' },
      { status: 500 }
    );
  }
}
