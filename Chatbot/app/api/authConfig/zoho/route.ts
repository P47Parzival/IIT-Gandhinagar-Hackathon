import { NextRequest, NextResponse } from 'next/server';
import { getComposio } from '../../../utils/composio';
import { createClient } from '@/app/utils/supabase/server';

// Zoho CRM Custom Auth Config  
const ZOHO_CRM_CONFIG = {
  name: 'zoho_crm_custom',
  toolkit: 'zoho', // Use 'zoho' (lowercase) to match actual connection data
  authType: 'oauth2',
  scopes: [
    'ZohoCRM.modules.ALL',
    'ZohoCRM.settings.ALL',
    'ZohoCRM.users.READ',
    'ZohoCRM.org.READ'
  ],
  authParams: {
    client_id: process.env.ZOHO_CLIENT_ID,
    client_secret: process.env.ZOHO_CLIENT_SECRET,
    redirect_uri: 'https://backend.composio.dev/api/v1/auth-apps/add',
    authorization_url: 'https://accounts.zoho.com/oauth/v2/auth',
    token_url: 'https://accounts.zoho.com/oauth/v2/token',
    refresh_url: 'https://accounts.zoho.com/oauth/v2/token'
  }
};

// GET: Get existing Zoho CRM auth config (DO NOT CREATE NEW ONES)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json(
        { error: 'Unauthorized - Please sign in' }, 
        { status: 401 }
      );
    }

    // Use the hardcoded existing auth config ID - DO NOT CREATE NEW ONES
    const EXISTING_ZOHO_AUTH_CONFIG_ID = 'ac_q9q5L3lpE9YS';
    
    console.log('Using existing Zoho CRM auth config:', EXISTING_ZOHO_AUTH_CONFIG_ID);

    return NextResponse.json({
      success: true,
      authConfig: {
        id: EXISTING_ZOHO_AUTH_CONFIG_ID,
        name: 'zoho_crm_existing',
        toolkit: 'zoho'
      },
      message: 'Using existing Zoho CRM auth config - no new configs created'
    });

  } catch (error) {
    console.error('Error accessing Zoho CRM auth config:', error);
    return NextResponse.json(
      { 
        error: 'Failed to access Zoho CRM auth config',
        details: error instanceof Error ? error.message : 'Unknown error'
      }, 
      { status: 500 }
    );
  }
}

// POST: Create connection link for Zoho CRM
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    // Check authentication
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user || !user.email) {
      return NextResponse.json(
        { error: 'Unauthorized - Please sign in' }, 
        { status: 401 }
      );
    }

    const composio = getComposio();

    if (action === 'connect') {
      // Use the existing Zoho CRM auth config ID - DO NOT CREATE NEW ONES
      const EXISTING_ZOHO_AUTH_CONFIG_ID = 'ac_q9q5L3lpE9YS';
      const COMPOSIO_USER_ID = 'pg-test-604fcca1-2eb4-4314-9907-75826a1622cd';
      
      console.log('Using existing auth config for connection:', {
        authConfigId: EXISTING_ZOHO_AUTH_CONFIG_ID,
        composioUserId: COMPOSIO_USER_ID,
        note: 'NO NEW AUTH CONFIGS WILL BE CREATED'
      });

      // Use initiate instead of link to prevent creating new auth configs
      const connectionRequest = await composio.connectedAccounts.initiate(
        COMPOSIO_USER_ID,
        EXISTING_ZOHO_AUTH_CONFIG_ID,
        {
          allowMultiple: true
        }
      );

      console.log('Zoho CRM connection link created for Composio user:', COMPOSIO_USER_ID);

      return NextResponse.json({
        success: true,
        connectionRequest,
        redirectUrl: connectionRequest.redirectUrl,
        message: 'Zoho CRM connection link created successfully'
      });
    }

    return NextResponse.json(
      { error: 'Invalid action. Use action: "connect"' },
      { status: 400 }
    );

  } catch (error) {
    console.error('Error creating Zoho CRM connection:', error);
    return NextResponse.json(
      { 
        error: 'Failed to create Zoho CRM connection',
        details: error instanceof Error ? error.message : 'Unknown error'
      }, 
      { status: 500 }
    );
  }
}
