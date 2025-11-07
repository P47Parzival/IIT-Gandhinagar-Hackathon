import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    // The official Composio callback endpoint per documentation  
    const composioCallbackUrl = 'https://backend.composio.dev/api/v1/auth-apps/add';

    // Extract and preserve all query parameters from the OAuth provider
    const queryParams = new URLSearchParams();
    request.nextUrl.searchParams.forEach((value, key) => {
      queryParams.append(key, value);
    });

    // Construct the redirect URL with all OAuth parameters
    const redirectUrl = `${composioCallbackUrl}?${queryParams.toString()}`;
    
    console.log('Forwarding OAuth callback to Composio:', {
      originalUrl: request.nextUrl.href,
      targetUrl: redirectUrl.substring(0, 150) + '...',
      paramsCount: queryParams.toString().length
    });

    // Redirect directly to Composio's callback endpoint
    // This maintains OAuth security while using custom domain for consent screen
    return NextResponse.redirect(redirectUrl);
    
  } catch (error) {
    console.error('Error in OAuth callback handler:', error);
    
    // Redirect to error page with details
    const errorUrl = new URL(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/`);
    errorUrl.searchParams.set('tab', 'chat');
    errorUrl.searchParams.set('error', 'oauth_callback_failed');
    errorUrl.searchParams.set('message', encodeURIComponent(
      error instanceof Error ? error.message : 'OAuth callback processing failed'
    ));
    
    return NextResponse.redirect(errorUrl.toString());
  }
}

export async function POST(request: NextRequest) {
  // Handle POST callbacks if needed
  return GET(request);
}
