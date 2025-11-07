# Zoho CRM Tool Router Integration Guide

This guide explains the complete Zoho CRM integration with Nirva AI assistant, including intelligent Tool Router detection and automatic OAuth flow.

## 🎯 How It Works

### **Intelligent Detection System**
Your Tool Router now automatically detects Zoho CRM queries and handles OAuth seamlessly:

1. **User asks**: *"Show me my Zoho CRM leads"*
2. **System detects**: Zoho CRM query keywords
3. **Checks connection**: User's Zoho CRM auth status
4. **If not connected**: Initiates OAuth flow automatically
5. **If connected**: Routes to Zoho CRM tools via Tool Router

### **Dual Integration Approach**
- **🤖 Chat Interface**: Automatic detection and OAuth prompting
- **⚙️ Link Data Tab**: Manual connection management for admins

---

## 🔧 Setup Instructions

### Step 1: Create Zoho API Application

1. **Access Zoho API Console**
   - Go to https://api-console.zoho.com/
   - Sign in with your Zoho account

2. **Create Server Application**
   - Click "Add Client" → "Server-based Applications"
   - Fill in the details:
     - **Client Name**: `Nirva AI Assistant`
     - **Homepage URL**: `http://localhost:3000` (or your domain)
     - **Redirect URI**: `http://localhost:3000/api/oauth/callback`

3. **Note Your Credentials**
   - After creation, copy your **Client ID** and **Client Secret**
   - Keep these secure - you'll need them for environment variables

### Step 2: Configure Environment Variables

Add these to your `.env.local` file:

```bash
# Zoho CRM Integration
ZOHO_CLIENT_ID=your_zoho_client_id_here
ZOHO_CLIENT_SECRET=your_zoho_client_secret_here

# Required for Composio integration
COMPOSIO_API_KEY=your_composio_api_key_here
```

### Step 3: Restart Development Server

```bash
# Stop current server (Ctrl+C)
npm run dev
```

---

## 🧪 Testing the Integration

### **Method 1: Chat Interface (Recommended)**

1. **Navigate to Chat Tab**
2. **Ask Zoho CRM Questions**:
   ```
   "Show me my Zoho CRM contacts"
   "What are my latest Zoho leads?"
   "Generate a report from Zoho deals"
   "Update contact status in Zoho CRM"
   ```

3. **First-Time Experience**:
   - System detects Zoho query
   - Displays OAuth connection prompt
   - Provides direct authorization link
   - User clicks → OAuth popup opens
   - After authorization → Returns to chat
   - Ask question again → Tool Router handles it

### **Method 2: Link Data Tab (Admin Management)**

1. **Navigate to Link Data Tab**
2. **Click "Add Connection"**
3. **Select Zoho CRM** (purple card with 🎯 icon)
4. **Fill connection details**
5. **Click "Test & Connect"**

---

## 🔍 Query Detection Keywords

The Tool Router detects these Zoho-related terms:
- `zoho crm`, `zoho leads`, `zoho contacts`
- `zoho deals`, `zoho accounts`, `zoho campaigns`
- `zoho opportunities`, `zoho modules`

### **Example Queries That Trigger Detection**:
✅ *"Show me Zoho CRM dashboard"*
✅ *"Export Zoho leads to spreadsheet"*
✅ *"What's my Zoho campaign performance?"*
❌ *"Show me leads"* (too generic - won't trigger)

---

## 🛠️ Technical Architecture

### **OAuth Flow Diagram**:
```
User Query → Zoho Detection → Connection Check
                                    ↓
             No Connection ← → Has Connection
                  ↓                    ↓
           OAuth Initiate        Tool Router
                  ↓                    ↓
           Authorization         Execute Action
                  ↓                    ↓
           Return to Chat       Return Results
```

### **API Endpoints**:
- **Chat Route**: `/api/chat` - Main Tool Router with Zoho detection
- **Auth Config**: `/api/authConfig/zoho` - Custom Zoho OAuth management
- **Generic Callback**: `/api/oauth/callback` - Universal OAuth handler
- **Connection Status**: `/api/connectedAccounts` - Check user connections

### **Scopes Requested**:
```javascript
[
  'ZohoCRM.modules.ALL',    // Access all CRM modules
  'ZohoCRM.settings.ALL',   // CRM configuration access
  'ZohoCRM.users.READ',     // User information
  'ZohoCRM.org.READ'        // Organization details
]
```

---

## 🔒 Security & Best Practices

### **OAuth Security**:
- ✅ OAuth2 secure authorization flow
- ✅ No passwords stored in system
- ✅ User-specific connection tokens
- ✅ Automatic token refresh handling
- ✅ Users can revoke access anytime

### **Environment Variables**:
```bash
# Development
NEXT_PUBLIC_APP_URL=http://localhost:3000
ZOHO_CLIENT_ID=dev_client_id
ZOHO_CLIENT_SECRET=dev_client_secret

# Production  
NEXT_PUBLIC_APP_URL=https://your-domain.com
ZOHO_CLIENT_ID=prod_client_id
ZOHO_CLIENT_SECRET=prod_client_secret
```

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**:

**🔴 "Environment variables not configured"**
```bash
# Check .env.local file exists and has:
ZOHO_CLIENT_ID=...
ZOHO_CLIENT_SECRET=...
```

**🔴 OAuth popup blocked**
- Enable popups for your domain
- Try incognito/private browsing mode

**🔴 "Redirect URI mismatch"**
- Ensure Zoho console has: `http://localhost:3000/api/oauth/callback`
- Check `NEXT_PUBLIC_APP_URL` matches your domain

**🔴 Tool Router not detecting queries**
- Use specific keywords: "Zoho CRM", "Zoho leads", etc.
- Avoid generic terms like just "leads" or "contacts"

### **Debug Mode**:
Check server logs for:
```bash
"Detected Zoho CRM query - checking connection status"
"No Zoho CRM connection found - initiating OAuth flow"  
"Zoho CRM connection found - proceeding with Tool Router"
```

---

## 🎉 Success Indicators

### **✅ Integration Working When**:
1. **Detection**: Zoho queries trigger custom OAuth flow
2. **OAuth**: Authorization popup opens correctly
3. **Connection**: User appears in connected accounts
4. **Tool Router**: Subsequent queries use Zoho tools
5. **Isolation**: Default toolkits (Slack, Google) work normally

### **🎯 Expected User Experience**:
```
User: "Show me Zoho CRM contacts"
→ System: "🎯 Zoho CRM Connection Required [OAuth Link]"
→ User: *Clicks link, authorizes*
→ User: "Show me Zoho CRM contacts" (again)
→ System: "📊 Here are your Zoho contacts: [Data]"
```

---

## 🚀 Next Steps

### **After Successful Integration**:
- Test various Zoho CRM queries
- Verify Tool Router routes correctly  
- Check other toolkits still work (Slack, Google Docs)
- Monitor connection status in Link Data tab
- Train users on natural language queries

### **Production Deployment**:
1. Update Zoho API Console with production URLs
2. Set production environment variables
3. Test OAuth flow on production domain
4. Monitor connection logs and errors

---

*🎯 This integration provides seamless Zoho CRM access through intelligent query detection while maintaining compatibility with all 500+ default Composio toolkits.*
