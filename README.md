# IIT Gandhinagar Hackathon

Bringing accountant-grade intelligence, tool automation, and anomaly detection together under a single workspace. This monorepo hosts two flagship applications that form the demo stack for the hackathon challenge:

1. **Nirva Chatbot** – a Next.js powered agent that can actually *do* work across 500+ business apps.
2. **Finnovate Platform** – a Django-based control centre for finance teams with RLHF-enabled chat, data linking, and governance dashboards.

Whether you want to try the autonomous agent, the finance command centre, or wire the two together, this README walks you through the full experience.

---

## 🚀 Projects at a Glance

| Project | Tech Stack | Highlights |
|---------|------------|------------|
| **Chatbot/** (`Nirva`) | Next.js 15, React 19, Tailwind, Supabase, Composio Tool Router | Agentic chat, 500+ SaaS integrations, OAuth bridge, streaming responses, MCP sessions |
| **finnovate_project/** (`Finnovate`) | Django 5, Tailwind (standalone), RLHF toolchain | Dual-response RLHF UI, SAP-style reports, Link Data wizard, internal ↔ external agent toggle |
| **Data_Processing/** | Python, PyTorch, Federated Learning | BalanceGuard anomaly detector & validator pipelines (supporting services) |

---

## 🏗️ Architecture Overview

### System Architecture

![System Architecture](assests\architecture_diagram.png)

### CFO Workflow

![CFO Workflow](assests\cfo_workflow.png)

The CFO workflow demonstrates the end-to-end process:
1. **Data Ingestion** – Import GL data from SAP/ERP systems
2. **AI Analysis** – Automated anomaly detection and validation
3. **Review & Approval** – RLHF-enabled chat interface for human oversight
4. **Report Generation** – Real-time dashboards and compliance reports
5. **Tool Integration** – Seamless connection to Slack, Zoho, and other business tools

* Flip a toggle in Finnovate to forward chat queries to the Nirva agent without leaving the Django UI.
* Composio handles authentication + tool execution, Supabase persists chat history, and Django captures RLHF comparisons.

---

## 📁 Repository Structure

```
├── Chatbot/                         # Nirva agentic chat (Next.js)
│   ├── app/api/chat/route.ts        # Tool Router + MCP session entrypoint
│   ├── app/api/authConfig/          # Composio auth config helpers
│   ├── app/components/              # Chat UI, tool panels, sidebars
│   └── README.md                    # Component-level documentation
│
├── finnovate_project/               # Django finance control centre
│   ├── fintech_project/core_APP/modules/
│   │   ├── dashboard/               # RLHF chat, tool router bridge
│   │   ├── link_data/               # GL uploads & SAP connectors
│   │   ├── reports/                 # SAP-style responsibility matrix
│   │   └── base/                    # Navbar, footer, shared assets
│   └── context.md                   # Detailed architecture walkthrough
│
├── Data_Processing/
│   ├── Anomaly_Detector/CONTEXT.md  # BalanceGuard federated learning notes
│   └── Anomaly_Validator/CONTEXT.md
│
├── assets/                          # Architecture diagrams and mock images
│
└── README.md (you are here)
```

---

## 📸 Application Screenshots

### Dashboard Interface
![Dashboard](assests\Dashboard.png)

The main dashboard provides:
- Real-time financial metrics and KPIs
- Anomaly detection alerts with severity indicators
- Quick access to recent GL entries and reviews
- Integration status with connected business tools


### Data Linking Module
![Link Data](assests\link_data.png)

Streamlined data integration:
- Upload wizard for GL and trial balance data
- SAP ERP connection configuration
- Real-time validation and error checking
- Supporting document attachment workflow

### RLHF Chat Interface
![Chat Interface](assests\anomaly-workflow.jpg)

Intelligent conversational AI:
- Dual-response comparison (internal vs. external agent)
- Tool execution visualization with real-time updates
- Context-aware financial query handling
- Seamless integration with Nirva chatbot for extended capabilities

---

## 🔧 Prerequisites

- Node.js 20+ and npm/yarn/pnpm for the Next.js app
- Python 3.11+, pipenv/virtualenv for Django & data pipelines
- Supabase project and credentials (for chat persistence)
- Composio API key (for tool integrations)
- OpenAI / Google Gemini keys (for LLM responses)

> **Tip:** Copy `Data_Processing/env_example.txt` when configuring the anomaly services.

---

## ⚙️ Environment Variables

### Chatbot (`Chatbot/.env.local`)

```env
COMPOSIO_API_KEY=...
OPENAI_API_KEY=...
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
NEXT_PUBLIC_APP_URL=http://localhost:3000
CHATBOT_INTERNAL_TOKEN=super-secret-token   # shared with Django bridge
```

### Finnovate (`finnovate_project/.env`)

```env
SECRET_KEY=replace-me
DATABASE_URL=sqlite:///db.sqlite3   # or Postgres
OPENAI_API_KEY=...
GOOGLE_AI_API_KEY=...
COMPOSIO_API_KEY=...
CHATBOT_API_BASE_URL=http://localhost:3000
CHATBOT_INTERNAL_TOKEN=super-secret-token  # must match Chatbot
```

---

## 🏃‍♀️ Running Everything Locally

1. **Install dependencies**
   ```bash
   # Chatbot
   cd Chatbot
   npm install

   # Finnovate
   cd ../finnovate_project
   pip install -r requirements.txt
   ```

2. **Apply Django migrations & collect static (optional)**
   ```bash
   cd finnovate_project/fintech_project
   python manage.py migrate
   python manage.py collectstatic --noinput
   ```

3. **Launch services in separate terminals**
   ```bash
   # Terminal A – Nirva agent
   cd Chatbot
   npm run dev

   # Terminal B – Finnovate dashboard
   cd finnovate_project/fintech_project
   python manage.py runserver
   ```

4. **Access the apps**
   - Chatbot UI: <http://localhost:3000>
   - Finnovate Dashboard: <http://localhost:8000/dashboard/>
   - Reports page: <http://localhost:8000/reports/>

5. **Bridge the agent (optional)**
   - Sign in to Finnovate, disable "Compare Mode", and ask a Zoho/Slack question.
   - Requests proxy to Next.js via `CHATBOT_INTERNAL_TOKEN`, so the agent runs inside the Django UI.

---

## ✨ Feature Highlights

### Nirva Chatbot

- Real-time streaming responses with tool execution callouts
- Automatic detection of Zoho CRM connections, fallback to existing auth configs
- Conversation caching via Supabase, Tool Router session reuse with MCP
- Custom Zoho integrations: `/api/authConfig/zoho`, `/app/auth/callback/zoho`

### Finnovate Platform

- Dual-response RLHF display with animated timelines and selection workflow
- "Compare Mode" toggle to swap between internal static model and external agent
- Link Data module with SAP ERP connection, upload wizard, and tailwind styling
- SAP-inspired Reports page showcasing a RACI responsibility matrix

### BalanceGuard Pipelines (Data_Processing)

- Federated + continual learning anomaly detector based on recent research
- Detailed notebooks & context files for validator/detector setup
- Ready-to-integrate parameters for EWC, Replay, FedAvg, and more

---

## 📚 Deep Dives

- `Chatbot/README.md` – Nirva agent architecture & Tool Router walkthrough
- `finnovate_project/context.md` – Full UI/UX + Tailwind + RLHF documentation
- `Data_Processing/Anomaly_Detector/CONTEXT.md` – BalanceGuard ML design

---

> Questions or demos? Ping the maintainers on table with team name D9THC
