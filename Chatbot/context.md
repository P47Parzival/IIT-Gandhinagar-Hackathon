# Nirva Chatbot - Project Context

## 📖 Project Overview

This is **Nirva**, an intelligent chatbot application built for financial operations automation, specifically focusing on **Automated Balance Sheet Assurance** for the IIT Gandhinagar Hackathon. The project leverages **Composio's Tool Router** and **Model Context Protocol (MCP)** to create an AI agent capable of interacting with 500+ applications and performing complex financial tasks.

### 🎯 Primary Purpose
- **Financial AI Assistant**: Specialized in General Ledger (GL) to Trial Balance (TB) conversion
- **Multi-Tool Integration**: Connects to Google Workspace, email, Slack, SAP, and more via Composio
- **Scalable Processing**: Handles large financial datasets with asynchronous job processing
- **CPA-Level Accuracy**: Enhanced with accounting domain expertise and validation

---

## 🏗️ Technical Architecture

### **Frontend Stack**
- **Next.js 15** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **React** with modern hooks and components
- **Supabase Client** for real-time data and authentication

### **Backend & AI Stack**
- **Google Gemini 2.5-Pro** as the primary LLM
- **Composio Tool Router** for intelligent tool discovery and execution
- **Model Context Protocol (MCP)** for AI-tool communication
- **Supabase PostgreSQL** for data persistence and storage
- **Next.js API Routes** for backend functionality

### **External Integrations**
- **Composio Platform**: 500+ application integrations
- **Google Workspace**: Sheets, Drive, Docs automation  
- **Communication**: Email, Slack messaging
- **File Processing**: CSV, Excel, Parquet data conversion
- **Storage**: Supabase Storage for file management

---

## ⚡ Key Features & Capabilities

### **1. Intelligent Chat Interface**
- **Contextual Conversations**: Maintains conversation history in Supabase
- **Dynamic UI**: Adaptive interface with Balance Sheet Panel integration
- **Real-time Processing**: WebSocket-like experience with streaming responses
- **Multi-modal Support**: Text, file uploads, and structured data interaction

### **2. Financial AI Assistant**
- **GL to TB Conversion**: Automated General Ledger to Trial Balance processing
- **CPA-Level Expertise**: Domain-specific prompting with accounting knowledge
- **Real-time Validation**: Accounting calculations verified during processing
- **Error Detection**: Identifies missing validations and accounting assumptions

### **3. Asynchronous Job Processing**
- **Background Workers**: Heavy TB processing decoupled from chat interface
- **Job Management**: Create, monitor, and retrieve processing jobs
- **Status Tracking**: Real-time progress updates and completion notifications
- **Scalable Architecture**: Handles millions of rows efficiently

### **4. Tool Router Integration**
- **Smart Tool Discovery**: Automatic selection of appropriate tools based on user intent
- **Function Calling**: Seamless integration with external APIs and services
- **Session Management**: Persistent MCP sessions for continuity
- **Fallback Handling**: Graceful degradation when external services are unavailable

---

## 📁 Project Structure

```
Chatbot/
├── app/
│   ├── api/                     # Backend API routes
│   │   ├── chat/               # Main chat processing endpoint
│   │   ├── tb-jobs/            # Trial Balance job management
│   │   │   ├── create/         # Job creation endpoint  
│   │   │   ├── [jobId]/        # Job status and results
│   │   │   └── process/        # Background worker
│   │   ├── conversations/      # Chat history management
│   │   ├── balance-sheet/      # Balance Sheet Panel API bridge
│   │   └── [other endpoints]   # Additional API routes
│   ├── components/             # React UI components
│   │   ├── ChatContainer.tsx   # Main chat interface
│   │   ├── ChatMessages.tsx    # Message display component
│   │   ├── BalanceSheetPanel.tsx # Financial insights panel
│   │   ├── MessageInput.tsx    # User input component
│   │   └── [ui components]     # Reusable UI elements
│   ├── hooks/                  # Custom React hooks
│   │   ├── useChatMessages.ts  # Chat state management
│   │   ├── useConversations.ts # Conversation management
│   │   └── useStreamingChat.ts # Real-time chat streaming
│   ├── utils/                  # Utility functions
│   │   ├── supabase/          # Database client configuration
│   │   ├── chat-history.ts    # Chat persistence logic
│   │   ├── composio.ts        # Composio integration
│   │   └── logger.ts          # Logging utilities
│   └── globals.css            # Global styles
├── database/                   # Database setup scripts
│   ├── tb_processing_schema.sql # TB job tables
│   └── storage_setup.sql      # Supabase Storage bucket
├── middleware.ts              # Next.js middleware
├── TB_PROCESSING_SETUP.md     # Setup instructions
└── context.md                 # This documentation file
```

---

## 🔌 API Endpoints

### **Chat & Conversations**
- `POST /api/chat` - Main chat processing with AI agent
- `GET /api/conversations` - Fetch user conversation history
- `GET /api/conversations/[id]/messages` - Get messages for specific conversation

### **Trial Balance Job Management**
- `POST /api/tb-jobs/create` - Create new TB processing job
- `GET /api/tb-jobs/[jobId]` - Get job status, progress, and results
- `POST /api/tb-jobs/process` - Background worker endpoint (internal)
- `GET /api/tb-jobs/[jobId]/export/[format]` - Download processed results

### **Balance Sheet Panel**
- `POST /api/balance-sheet` - Financial analysis queries
- `GET /api/balance-sheet` - Get analysis suggestions

### **Authentication & Management**
- `GET /api/authConfig/all` - Composio authentication configuration
- `GET /api/authLinks` - Get authorization links for tools
- `GET /api/connectedAccounts` - Manage connected external accounts

---

## 🗄️ Database Schema

### **Core Tables (Supabase PostgreSQL)**
```sql
-- Conversations and Messages
conversations (id, user_id, title, created_at, updated_at)
messages (id, conversation_id, content, role, created_at, metadata)

-- TB Job Processing
tb_jobs (id, user_id, status, progress, source, metadata, created_at, updated_at)
tb_aggregates (id, job_id, account_code, account_name, total_debit, total_credit, net_balance)
tb_summary (id, job_id, total_debits, total_credits, is_balanced, data_quality_score)
```

### **Storage Buckets**
- `tb-processing-files` - Raw CSV uploads and processed Parquet files
- Row Level Security (RLS) enabled for user data isolation

---

## 🔧 Configuration & Environment

### **Required Environment Variables**
```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Composio Configuration  
COMPOSIO_API_KEY=your_composio_api_key
COMPOSIO_BASE_URL=https://backend.composio.dev

# AI Model Configuration
GOOGLE_API_KEY=your_google_ai_api_key
```

### **Key Dependencies**
- `@ai-sdk/google` - Google AI integration
- `@supabase/ssr` - Supabase Server-Side Rendering
- `composio-core` - Composio SDK for tool integration
- `ai` - AI SDK for streaming and function calling
- `next` - Next.js framework
- `react` - React library

---

## 🧠 AI Agent Architecture

### **Model Configuration**
- **Primary Model**: Google Gemini 2.5-Pro
- **Temperature**: Dynamic (0.1 for accounting, 0.3 for general)
- **Streaming**: Enabled for real-time responses
- **Function Calling**: Integrated with Composio Tool Router

### **System Prompt Enhancement**
```typescript
// Accounting-specific expertise injection
const accountingPrompt = `
You are a CPA-certified financial expert with deep knowledge of:
- Generally Accepted Accounting Principles (GAAP)
- International Financial Reporting Standards (IFRS)  
- Trial balance preparation and validation
- GL account classification and analysis
- Financial statement preparation
- Audit and assurance procedures
`;
```

### **Real-time Validation**
- **onStepFinish**: Validates accounting calculations during processing
- **Error Detection**: Identifies missing validations and assumptions
- **Quality Assurance**: Ensures CPA-level accuracy in financial operations

---

## 🔄 TB Processing Workflow

### **Asynchronous Job Pipeline**
1. **Job Creation**: User requests TB conversion → Job created in `tb_jobs` table
2. **Background Processing**: 
   - Download CSV from Google Sheets
   - Store raw data in Supabase Storage
   - Convert CSV to Parquet for efficient processing
   - Run SQL aggregation in PostgreSQL
   - Store results in `tb_aggregates` and `tb_summary` tables
3. **Completion**: Job marked as complete, download links generated
4. **User Notification**: Status available via API or chat query

### **Data Transformation Flow**
```
Google Sheets → CSV Download → Supabase Storage → 
Parquet Conversion → PostgreSQL Aggregation → Trial Balance Report
```

---

## 🎨 UI Components Architecture

### **Dynamic Layout System**
- **ChatContainer**: Main orchestrator with conditional panel display
- **Two-Panel Layout**: Chat interface + Balance Sheet Panel (when triggered)
- **Responsive Design**: Adapts to different screen sizes and use cases

### **State Management**
- **React Hooks**: Custom hooks for chat, conversations, and streaming
- **Local State**: Component-level state for UI interactions  
- **Supabase Real-time**: Database-driven state updates
- **Session Persistence**: MCP sessions maintained across interactions

### **User Experience Features**
- **Welcome Screen**: Quick action buttons for common tasks
- **Message Streaming**: Real-time AI responses with typing indicators
- **File Upload Support**: Drag-and-drop for financial documents
- **Export Options**: Multiple formats (CSV, Excel) for processed data

---

## 🔗 Integration Points

### **External System Connections**
- **Balance Sheet Assurance Backend**: FastAPI service for financial analysis
- **Data Processing Module**: Anomaly detection and ML workflows
- **Integrated AI Agent**: Unified agent orchestrating multiple capabilities

### **Composio Tool Categories**
- **Productivity**: Google Workspace (Sheets, Docs, Drive)
- **Communication**: Email (Gmail), Slack, Microsoft Teams
- **File Management**: Document generation, data export
- **Financial Systems**: SAP integration capabilities (planned)

---

## 📈 Performance Optimizations

### **Scalability Features**
- **Asynchronous Processing**: Non-blocking TB generation for large datasets
- **Database Optimization**: Indexed queries and efficient data structures
- **Caching Strategy**: MCP session reuse and tool router optimization
- **Error Handling**: Graceful fallbacks and retry mechanisms

### **Code Quality Standards**
- **TypeScript**: Full type safety across the application
- **Linting**: ESLint configuration for code consistency
- **Error Boundaries**: Robust error handling and user feedback
- **Logging**: Structured logging with different severity levels

---

## 🚀 Recent Enhancements

### **Phase 1: Accuracy Improvements (Completed)**
- ✅ Enhanced system prompt with CPA-level accounting expertise
- ✅ Dynamic temperature control (0.1 for accounting, 0.3 for general)
- ✅ Real-time accounting validation in `onStepFinish`
- ✅ Error detection and assumption logging

### **Phase 2: Scalability Architecture (Completed)**
- ✅ Asynchronous TB job processing system
- ✅ Background worker for heavy computations  
- ✅ Database schema for job management
- ✅ Supabase Storage integration for file handling
- ✅ Production-ready error handling and logging

### **Bug Fixes & Improvements**
- ✅ Fixed Supabase client initialization issues
- ✅ Resolved circular import problems
- ✅ Enhanced Balance Sheet Panel query routing
- ✅ Improved fallback mechanisms for external service unavailability

---

## 🎯 Use Cases & Examples

### **Primary Use Cases**
1. **GL to TB Conversion**: "Convert my general ledger from Google Sheet 'GL' into a trial balance report"
2. **Financial Analysis**: "Analyze the balance sheet variances for Q3"
3. **Document Generation**: "Create an audit report based on the trial balance"
4. **Communication Automation**: "Email the CFO about the monthly reconciliation status"
5. **Job Status Monitoring**: "What's the status of job abc123-def456?"

### **Multi-Tool Workflows**
- **Data Collection**: Google Sheets → Processing → Storage → Analysis
- **Report Generation**: Analysis → Document Creation → Email Distribution
- **Compliance Checking**: Data Validation → Audit Trail → Management Reporting

---

## 🔮 Future Development Areas

### **Planned Enhancements**
- **Advanced Analytics**: ML-powered anomaly detection integration
- **SAP Integration**: Direct ERP connectivity for enterprise clients
- **Multi-Entity Support**: Consolidated reporting across business units
- **Advanced Visualizations**: Interactive financial dashboards
- **Audit Trail**: Comprehensive compliance and audit logging

### **Technical Roadmap**
- **Performance**: Further optimization for enterprise-scale datasets
- **Security**: Enhanced authentication and authorization
- **Monitoring**: Application performance monitoring and alerting
- **API**: RESTful API expansion for third-party integrations

---

## 📞 Key Integration Commands

### **For AI Assistants Working on This Project**

**Understanding the Chat Flow:**
```typescript
// Main chat endpoint handles both regular chat and TB job routing
POST /api/chat → Detects TB requests → Creates job OR processes normally
```

**Working with TB Jobs:**
```typescript
// Create job: POST /api/tb-jobs/create
// Check status: GET /api/tb-jobs/[jobId]  
// Background processing: POST /api/tb-jobs/process
```

**Database Operations:**
```typescript
// Use Supabase client: const supabase = await createClient()
// Always handle RLS: User authentication required for data access
// Job tracking: tb_jobs table for metadata, tb_aggregates for results
```

**UI Component Integration:**
```typescript
// ChatContainer orchestrates the entire interface
// BalanceSheetPanel shows financial insights
// Dynamic routing based on user query patterns
```

---

This context file provides a comprehensive overview of the Nirva Chatbot project, its architecture, and capabilities. It should enable other AI assistants to quickly understand and contribute to the project effectively.
  
**Project Status**: Production Ready with Advanced Financial Processing Capabilities
