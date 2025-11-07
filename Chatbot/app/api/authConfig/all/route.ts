import { NextRequest, NextResponse } from 'next/server';
import { getComposio } from '../../../utils/composio';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const { includeCustom = false } = body;

    const composio = getComposio();
    
    // Always get the default Composio auth configs
    const defaultAuthConfigs = await composio.authConfigs.list();
    
    let allConfigs = defaultAuthConfigs;

    // If specifically requested, also include custom auth configs (like Zoho CRM)
    if (includeCustom) {
      try {
        // Check if Zoho CRM custom auth config exists
        // Attempt to fetch Zoho custom auth configs using known toolkit slugs
        let zohoConfigs = await composio.authConfigs.list({
          toolkit: 'zoho'
        });

        if (!zohoConfigs.items || zohoConfigs.items.length === 0) {
          zohoConfigs = await composio.authConfigs.list({
            toolkit: 'zoho_crm'
          });
        }

        if (zohoConfigs.items && zohoConfigs.items.length > 0) {
          // Merge Zoho CRM configs with default configs
          allConfigs = {
            ...defaultAuthConfigs,
            items: [
              ...(defaultAuthConfigs.items || []),
              ...zohoConfigs.items
            ]
          };
          console.log('Added Zoho CRM custom auth configs to response');
        }
      } catch (zohoError) {
        console.log('No Zoho CRM custom auth configs found or error fetching them');
      }
    }

    console.log(`Returning ${allConfigs.items?.length || 0} auth configs (including custom: ${includeCustom})`);
    return NextResponse.json(allConfigs);
  } catch (error) {
    console.error('Error fetching all auth configs:', error);
    return NextResponse.json(
      { error: 'Failed to fetch auth configs' }, 
      { status: 500 }
    );
  }
}