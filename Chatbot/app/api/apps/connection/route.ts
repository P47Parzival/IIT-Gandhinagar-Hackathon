import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/app/utils/supabase/server';
import { getComposio } from '@/app/utils/composio';

// GET: Check connection status for all toolkits for authenticated user
export async function GET(request: NextRequest) {
  try {
    // Get authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user || !user.email) {
      return NextResponse.json(
        { error: 'Unauthorized - Please sign in' }, 
        { status: 401 }
      );
    }

    const composio = getComposio();
    
    // Use the correct Composio User ID to find existing connections
    const COMPOSIO_USER_ID = 'pg-test-604fcca1-2eb4-4314-9907-75826a1622cd';
    
    // Fetch connected accounts for the correct Composio user
    const connectedAccounts = await composio.connectedAccounts.list({
      userIds: [COMPOSIO_USER_ID]
    });

    console.log('Connected accounts for Composio user:', COMPOSIO_USER_ID, `(${connectedAccounts.items?.length || 0} accounts)`);

    // Get detailed info for each connected account
    const detailedAccounts = await Promise.all(
      (connectedAccounts.items || []).map(async (account) => {
        try {
          const accountDetails = await composio.connectedAccounts.get(account.id);
          // Log only essential info without sensitive data
          console.log('Account details for', account.id, ':', {
            toolkit: accountDetails.toolkit?.slug,
            connectionId: accountDetails.id,
            authConfigId: accountDetails.authConfig?.id,
            status: accountDetails.status
          });
          return accountDetails;
        } catch (error) {
          console.error('Error fetching account details for', account.id, ':', error);
          return account; // fallback to original if details fetch fails
        }
      })
    );
    
    return NextResponse.json({ connectedAccounts: detailedAccounts });
  } catch (error) {
    console.error('Error fetching connection status:', error);
    return NextResponse.json(
      { error: 'Failed to fetch connection status' }, 
      { status: 500 }
    );
  }
}

// POST: Create auth link for connecting a toolkit
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { authConfigId, toolkitSlug } = body;
    
    if (!authConfigId) {
      return NextResponse.json(
        { error: 'authConfigId is required' }, 
        { status: 400 }
      );
    }

    // Get authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user || !user.email) {
      return NextResponse.json(
        { error: 'Unauthorized - Please sign in' }, 
        { status: 401 }
      );
    }

    console.log('Creating auth link for user:', user.email, 'toolkit:', toolkitSlug);

    // Special handling for Zoho CRM
    if (toolkitSlug && (toolkitSlug.toLowerCase().includes('zoho') || toolkitSlug === 'zoho_crm')) {
      console.log('Detected Zoho CRM connection request - checking existing connections first');
      
      try {
        // Check if user already has an ACTIVE Zoho connection - don't create new ones
        const COMPOSIO_USER_ID = 'pg-test-604fcca1-2eb4-4314-9907-75826a1622cd';
        const composio = getComposio();
        
        const existingConnections = await composio.connectedAccounts.list({
          userIds: [COMPOSIO_USER_ID]
        });
        
        const activeZohoConnection = existingConnections.items?.find(account => 
          account.toolkit?.slug === 'zoho' && account.status === 'ACTIVE'
        );
        
        if (activeZohoConnection) {
          console.log('ACTIVE Zoho connection already exists - not creating new one', {
            existingConnectionId: activeZohoConnection.id,
            status: activeZohoConnection.status
          });
          
          return NextResponse.json({
            redirectUrl: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}?tab=chat&message=Zoho%20CRM%20already%20connected`,
            isCustomAuth: true,
            toolkit: 'zoho',
            existingConnection: activeZohoConnection.id,
            note: 'Using existing ACTIVE connection - no new connection created'
          });
        }
        
        // Only create new connection if no ACTIVE one exists
        console.log('No ACTIVE Zoho connection found - proceeding with new connection');
        
        // Use initiate directly instead of creating new auth configs
        const EXISTING_ZOHO_AUTH_CONFIG_ID = 'ac_q9q5L3lpE9YS';
        
        console.log('Using existing Zoho auth config directly', {
          authConfigId: EXISTING_ZOHO_AUTH_CONFIG_ID,
          composioUserId: COMPOSIO_USER_ID
        });
        
        const connectionRequest = await composio.connectedAccounts.initiate(
          COMPOSIO_USER_ID,
          EXISTING_ZOHO_AUTH_CONFIG_ID,
          {
            allowMultiple: true
          }
        );

        console.log('Zoho CRM connection initiated directly with existing auth config');
        
        return NextResponse.json({
          redirectUrl: connectionRequest.redirectUrl,
          connectionId: connectionRequest.id,
          isCustomAuth: true,
          toolkit: 'zoho'
        });
        
      } catch (zohoError) {
        console.error('Error with Zoho CRM custom auth, falling back to default:', zohoError);
        // Fall back to default Composio handling if custom auth fails
      }
    }

    // Default handling for all other toolkits (Slack, Google Docs, etc.)
    // Note: For non-Zoho toolkits, we still use user.email as the Composio User ID
    const composio = getComposio();
    const connectionRequest = await composio.connectedAccounts.link(
      user.email, 
      authConfigId, 
      {
        callbackUrl: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/apps`
      }
    );
    
    console.log('Default auth link created for toolkit:', toolkitSlug);
    return NextResponse.json(connectionRequest);
    
  } catch (error) {
    console.error('Error creating auth link:', error);
    return NextResponse.json(
      { error: 'Failed to create auth link' }, 
      { status: 500 }
    );
  }
}

// DELETE: Disconnect a toolkit
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { accountId } = body;
    
    if (!accountId) {
      return NextResponse.json(
        { error: 'accountId is required' }, 
        { status: 400 }
      );
    }

    // Get authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json(
        { error: 'Unauthorized - Please sign in' }, 
        { status: 401 }
      );
    }

    console.log('Disconnecting account:', accountId, 'for user:', user.email);

    const composio = getComposio();
    const result = await composio.connectedAccounts.delete(accountId);
    
    console.log('Disconnect result:', result);
    return NextResponse.json({ success: true, result });
  } catch (error) {
    console.error('Error disconnecting account:', error);
    return NextResponse.json(
      { error: 'Failed to disconnect account' }, 
      { status: 500 }
    );
  }
}