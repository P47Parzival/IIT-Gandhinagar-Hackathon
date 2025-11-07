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

## 🧠 Architecture Overview

```
                             ┌────────────────────────────┐
                             │  Composio Tool Router & MCP │
                             └──────────────┬─────────────┘
                                            │
            ┌─────────────────────┐         │         ┌────────────────────┐
            │  Nirva Chatbot (UI) │◀────────┴────────▶│ External SaaS Apps │
            │  Next.js / Supabase │  Tool calls       │ Slack, Zoho, etc.  │
            └──────────┬──────────┘                     └────────────────────┘
                       │ REST bridge (internal token)
                       ▼
         ┌──────────────────────────────┐
         │ Finnovate Django Dashboard   │
         │ RLHF Chat • Reports • Link   │
         └──────────┬───────────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │ BalanceGuard Data Pipelines│
       │ (Federated Anomaly Models) │
       └────────────────────────────┘
```

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
└── README.md (you are here)
```

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
   python manage.py migrate
   python manage.py collectstatic --noinput
   ```

3. **Launch services in separate terminals**
   ```bash
   # Terminal A – Nirva agent
   cd Chatbot
   npm run dev

   # Terminal B – Finnovate dashboard
   cd finnovate_project
   python manage.py runserver
   ```

4. **Access the apps**
   - Chatbot UI: <http://localhost:3000>
   - Finnovate Dashboard: <http://localhost:8000/dashboard/>
   - Reports page: <http://localhost:8000/reports/>

5. **Bridge the agent (optional)**
   - Sign in to Finnovate, disable “Compare Mode”, and ask a Zoho/Slack or anything realted question.
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
- “Compare Mode” toggle to swap between internal static model and external agent
- Link Data module with SAP ERP connection, upload wizard, and tailwind styling
- SAP-inspired Reports page showcasing a RACI responsibility matrix

### BalanceGuard Pipelines (Data_Processing)

- Federated + continual learning anomaly detector based on recent research
- Detailed notebooks & context files for validator/detector setup
- Ready-to-integrate parameters for EWC, Replay, FedAvg, and more

---

## 🧪 Testing & QA

- **Chatbot**: `npm run lint` and `npm run build` to check TypeScript & Next.js.
- **Finnovate**: `python manage.py test` for Django unit tests (add your own).
- **Data Pipelines**: Follow the instructions inside each `CONTEXT.md` for reproducing experiments.

---

## 📚 Deep Dives

- `Chatbot/README.md` – Nirva agent architecture & Tool Router walkthrough
- `finnovate_project/context.md` – Full UI/UX + Tailwind + RLHF documentation
- `Data_Processing/Anomaly_Detector/CONTEXT.md` – BalanceGuard ML design

---

## 🤝 Contributing

Please DON'T

---

## 📜 License & Credits

> Questions or demos? Reach out on our table, team-name: **D9THC**
