import { NextRequest, NextResponse } from 'next/server';
import { getComposio } from '@/app/utils/composio';
import { createClient } from '@/app/utils/supabase/server';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const code = searchParams.get('code');
    const state = searchParams.get('state');
    const error = searchParams.get('error');

    // Handle OAuth errors
    if (error) {
      console.error('Zoho OAuth error:', error);
      const errorDescription = searchParams.get('error_description') || 'Unknown error';
      
      return NextResponse.redirect(
        `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/apps?error=zoho_auth_failed&message=${encodeURIComponent(errorDescription)}`
      );
    }

    if (!code) {
      return NextResponse.redirect(
        `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/apps?error=missing_code`
      );
    }

    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user || !user.email) {
      return NextResponse.redirect(
        `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/auth?error=unauthorized`
      );
    }

    console.log('Processing Zoho OAuth callback for user:', user.email);

    const composio = getComposio();

    try {
      // The callback flow for Composio custom auth configs
      // We need to forward the callback to Composio's endpoint with all query parameters
      
      // Extract all query parameters from the original request
      const allParams = new URLSearchParams();
      request.nextUrl.searchParams.forEach((value, key) => {
        allParams.append(key, value);
      });

      // Composio's callback endpoint for custom auth configs
      const composioCallbackUrl = 'https://backend.composio.dev/api/v1/auth-apps/add';
      const redirectUrl = `${composioCallbackUrl}?${allParams.toString()}`;
      
      console.log('Forwarding Zoho OAuth callback to Composio:', redirectUrl);

      // Make the callback request to Composio
      const callbackResponse = await fetch(redirectUrl, {
        method: 'GET',
        headers: {
          'X-API-KEY': process.env.COMPOSIO_API_KEY || '',
          'User-Agent': 'Nirva-Agent/1.0'
        }
      });

      if (!callbackResponse.ok) {
        throw new Error(`Composio callback failed: ${callbackResponse.status} ${callbackResponse.statusText}`);
      }

      const callbackResult = await callbackResponse.json();
      console.log('Zoho OAuth callback completed via Composio:', callbackResult);

      // Check if we can get the connection info
      let connectionId = null;
      if (callbackResult.connectedAccountId) {
        connectionId = callbackResult.connectedAccountId;
      } else {
        // Try to find the connection by querying connected accounts using correct Composio User ID
        try {
          const COMPOSIO_USER_ID = 'pg-test-604fcca1-2eb4-4314-9907-75826a1622cd';
          const connectedAccounts = await composio.connectedAccounts.list({
            userIds: [COMPOSIO_USER_ID]
          });
          const zohoConnection = connectedAccounts.items?.find(account => 
            account.toolkit?.slug?.toLowerCase().includes('zoho')
          );
          if (zohoConnection) {
            connectionId = zohoConnection.id;
          }
        } catch (listError) {
          console.warn('Could not verify connection:', listError);
        }
      }

      // Redirect back to the app with success indicators
      const successUrl = new URL(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/`);
      successUrl.searchParams.set('tab', 'chat');
      successUrl.searchParams.set('connected', 'zoho_crm');
      if (connectionId) {
        successUrl.searchParams.set('account_id', connectionId);
      }
      successUrl.searchParams.set('message', 'Zoho CRM connected successfully! You can now ask questions about your Zoho data.');

      return NextResponse.redirect(successUrl.toString());

    } catch (callbackError) {
      console.error('Error completing Zoho OAuth callback:', callbackError);
      
      const errorUrl = new URL(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/apps`);
      errorUrl.searchParams.set('error', 'callback_failed');
      errorUrl.searchParams.set('message', encodeURIComponent(
        callbackError instanceof Error ? callbackError.message : 'Failed to complete Zoho connection'
      ));

      return NextResponse.redirect(errorUrl.toString());
    }

  } catch (error) {
    console.error('Unexpected error in Zoho callback:', error);
    
    return NextResponse.redirect(
      `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/apps?error=callback_error`
    );
  }
}
