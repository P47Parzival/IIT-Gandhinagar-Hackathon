# 🎯 Zoho CRM Custom Integration Setup

This document explains how to set up the custom Zoho CRM integration that works alongside Composio's default toolkits.

## 📋 Overview

The Zoho CRM integration is implemented as a **custom auth config** that:
- ✅ **Only triggers for Zoho CRM requests**
- ✅ **Keeps all default Composio toolkits working** (Slack, Google Docs, etc.)
- ✅ **Uses custom OAuth2 flow** for Zoho CRM authentication
- ✅ **Seamlessly integrates** with the existing LinkData dashboard

---

## 🔧 Setup Instructions

### Step 1: Create Zoho API Application

1. Go to [Zoho API Console](https://api-console.zoho.com/)
2. Sign in with your Zoho account
3. Click **"Add Client"** → **"Server-based Applications"**
4. Fill in application details:
   - **Client Name**: `Nirva AI Assistant`
   - **Homepage URL**: `https://your-app-domain.com`
   - **Authorized Redirect URIs**: 
     ```
     http://localhost:3000/auth/callback/zoho
     https://your-app-domain.com/auth/callback/zoho
     ```

5. Note down your **Client ID** and **Client Secret**

### Step 2: Configure Environment Variables

Add these to your `.env.local` file:

```bash
# Zoho CRM Custom Integration
ZOHO_CLIENT_ID=your_zoho_client_id_here
ZOHO_CLIENT_SECRET=your_zoho_client_secret_here
```

### Step 3: Test the Integration

1. Navigate to the **"Link Data"** tab in your Nirva app
2. Click **"Add Connection"**
3. Select **"Zoho CRM"** (marked with 🎯 icon)
4. Fill in connection details and click **"Test & Connect"**
5. Complete OAuth flow in the popup window

---

## 🏗️ Technical Architecture

### Smart Detection Logic

The system automatically detects Zoho CRM requests:

```typescript
// Only triggers for Zoho CRM
if (toolkitSlug && (toolkitSlug.toLowerCase().includes('zoho') || toolkitSlug === 'zoho_crm')) {
  // Use custom auth config
} else {
  // Use default Composio auth configs
}
```

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/authConfig/zoho` | Get/create Zoho CRM auth config |
| `POST /api/authConfig/zoho` | Create Zoho CRM connection |
| `GET /auth/callback/zoho` | Handle OAuth callback |

### Default Toolkits Preserved

All existing Composio integrations continue to work normally:
- ✅ Slack
- ✅ Google Workspace (Docs, Sheets, Drive)
- ✅ GitHub
- ✅ Email (Gmail)
- ✅ 500+ other Composio toolkits

---

## 🔍 Troubleshooting

### Common Issues

**1. "Setup Required" Error**
```
Solution: Add ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET to .env.local
```

**2. "Redirect URI Mismatch" Error**
```
Solution: Ensure redirect URI in Zoho console matches your app URL:
- Local: http://localhost:3000/auth/callback/zoho
- Production: https://yourdomain.com/auth/callback/zoho
```

**3. "Connection Failed" Error**
```
Solution: Check Zoho API console for valid credentials and permissions
```

### Debug Logging

Check browser console and server logs for detailed error messages:

```bash
# Server logs will show:
"Detected Zoho CRM connection request - using custom auth config"
"Zoho CRM connection created via custom auth config"
```

---

## 🎯 Usage Examples

### Connect via Link Data Dashboard

1. Go to **Link Data** tab
2. Click **Add Connection**
3. Select **Zoho CRM** 🎯
4. Enter connection name: `"My Zoho CRM"`
5. Click **Test & Connect**
6. Authorize in OAuth popup

### Use in Chat

Once connected, you can query Zoho CRM via chat:

```
"Show me leads from Zoho CRM created this month"
"Update contact status in Zoho CRM for John Doe"
"Generate report from Zoho CRM data"
```

---

## 🔐 Security Features

- **OAuth2 Flow**: Secure authorization without storing passwords
- **Scoped Access**: Limited to necessary CRM permissions
- **Token Management**: Automatic token refresh handled by Composio
- **Isolation**: Custom auth doesn't affect other integrations

---

## 🚀 Production Deployment

### Environment Variables

```bash
# Production .env
ZOHO_CLIENT_ID=your_production_client_id
ZOHO_CLIENT_SECRET=your_production_client_secret
NEXT_PUBLIC_APP_URL=https://your-production-domain.com
```

### Zoho Console Settings

Update Zoho API console with production URLs:
- **Homepage URL**: `https://your-production-domain.com`
- **Redirect URI**: `https://your-production-domain.com/auth/callback/zoho`

---

## ✅ Verification

The integration is working correctly when:

1. **Link Data tab shows Zoho CRM** option with 🎯 icon
2. **Default toolkits still work** (Slack, Google Docs, etc.)
3. **Zoho CRM connection** opens OAuth popup
4. **Chat can access Zoho CRM** data after connection
5. **Server logs show** custom auth config usage

---

## 🎉 Success!

Your Nirva AI Assistant now supports:
- **Custom Zoho CRM integration** with OAuth2
- **All default Composio toolkits** working normally  
- **Intelligent routing** between custom and default auth configs
- **Professional UI** for connection management

The system seamlessly handles both custom integrations and Composio's extensive toolkit library! 🚀
