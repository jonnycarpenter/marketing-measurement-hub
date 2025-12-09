# Marketing Measurement Hub

A production-ready multi-agent AI application for designing marketing experiments and measuring incremental impact using causal inference.

**🔗 Live Demo:** [app.ketzeroconsulting.ai](https://app.ketzeroconsulting.ai)

---

## 🎯 What It Does

This application automates the end-to-end marketing measurement workflow that typically requires a data science team:

1. **Test Design** → AI agent designs statistically valid experiments (geo-splits, audience stratification, power analysis)
2. **Conflict Detection** → Checks against existing tests and promo calendar to prevent contamination
3. **Measurement** → Runs Bayesian CausalImpact analysis to measure true incremental lift
4. **Reporting** → Generates executive summaries with confidence intervals and go/no-go recommendations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │  Agents  │  │   Test   │  │  Master  │  │     Dataset      ││
│  │   Chat   │  │ Tracker  │  │   Test   │  │     Explorer     ││
│  └────┬─────┘  └──────────┘  │  Detail  │  └──────────────────┘│
│       │                      └──────────┘                       │
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Google Agent Development Kit (ADK)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Lira Maven    │  │   Ketin Vale    │  │  Brena Colette  │ │
│  │  (Lead Agent)   │──│  (Test Design)  │──│  (Measurement)  │ │
│  │                 │  │                 │  │                 │ │
│  │ • Routing       │  │ • DMA selection │  │ • CausalImpact  │ │
│  │ • Context mgmt  │  │ • Power analysis│  │ • Diagnostics   │ │
│  │ • Handoffs      │  │ • Conflict check│  │ • Reporting     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                              │                      │           │
│                              ▼                      ▼           │
│                    ┌─────────────────────────────────────────┐ │
│                    │              Agent Tools                │ │
│                    │  • query_data()     • save_test()       │ │
│                    │  • create_chart()   • run_measurement() │ │
│                    │  • search_kb()      • validate_split()  │ │
│                    └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   GCS Data  │  │  ChromaDB   │  │   tfp-causalimpact      │ │
│  │  (CSV/JSON) │  │    (RAG)    │  │  (Bayesian inference)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🤖 The Agents

| Agent | Role | Key Capabilities |
|-------|------|------------------|
| **Lira Maven** | Measurement Lead | Routes requests, manages context, orchestrates handoffs between specialists |
| **Ketin Vale** | Test Design Specialist | Designs geo-experiments, validates audience splits, checks for test conflicts |
| **Brena Colette** | Measurement Analyst | Runs CausalImpact models, interprets statistical results, generates reports |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **AI Orchestration** | Google ADK (Agent Development Kit) |
| **LLM** | Gemini 2.5 Pro / Flash (via Vertex AI) |
| **Causal Inference** | tfp-causalimpact (TensorFlow Probability) |
| **Vector Store** | ChromaDB (for RAG knowledge base) |
| **Data Storage** | Google Cloud Storage |
| **Deployment** | Cloud Run (containerized) |
| **Scheduling** | Cloud Scheduler + Cloud Functions (data purge) |

## 📁 Project Structure

```
mktg_measurement_streamline/
├── app.py                      # Main Streamlit application (3300+ lines)
├── agents/
│   ├── measurement_lead.py     # Lira - orchestration agent
│   ├── test_design_agent.py    # Ketin - experiment design
│   ├── measurement_agent.py    # Brena - causal analysis
│   ├── tools.py                # Shared agent tools
│   └── agent_prompts/          # Agent persona definitions
├── utils/
│   ├── causal_impact_utils.py  # CausalImpact wrapper
│   ├── data_loader.py          # GCS/local data abstraction
│   ├── gcs_loader.py           # Cloud Storage operations
│   ├── rag_utils.py            # Knowledge base retrieval
│   ├── test_design_utils.py    # DMA selection, power analysis
│   └── validators.py           # Input validation
├── knowledge_base/             # RAG documents (causal inference best practices)
├── cloud_functions/
│   └── purge_test_data/        # Daily data reset function
├── data/                       # Sample datasets
├── configs/                    # Agent configurations
├── Dockerfile                  # Container definition
├── cloudbuild.yaml             # CI/CD pipeline
└── requirements.txt            # Python dependencies
```

## 🚀 Local Development

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/marketing-measurement-hub.git
cd marketing-measurement-hub

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run locally
streamlit run app.py --server.port 8080
```

## ☁️ Deployment (Cloud Run)

```bash
# Deploy to Cloud Run
gcloud run deploy mktg-measurement-hub \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=your-project,GOOGLE_CLOUD_LOCATION=us-central1,GOOGLE_GENAI_USE_VERTEXAI=true"
```

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | GCP region (e.g., `us-central1`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | Set to `true` for Vertex AI auth |
| `GCS_BUCKET_NAME` | Cloud Storage bucket for data |

## 📊 Key Features

- **Multi-Agent Orchestration** - Agents collaborate via Google ADK with structured handoffs
- **RAG Knowledge Base** - Retrieves causal inference best practices during design
- **Real-time Data Queries** - Agents can query datasets to inform recommendations  
- **Automated Artifact Generation** - Test configs, audience files, and reports saved to GCS
- **Caching Strategy** - Streamlit caching for data + 5-min TTL for explorer datasets
- **Daily Data Purge** - Cloud Scheduler resets demo data nightly

## 📈 Sample Workflow

1. User: *"I want to test a 20% YouTube budget increase"*
2. **Lira** routes to **Ketin** for test design
3. **Ketin** queries promo calendar, checks for conflicts, recommends DMAs
4. User approves design → **Ketin** saves test config + audience files
5. After test runs, user asks **Brena** to measure results
6. **Brena** runs CausalImpact, generates diagnostics + executive summary

---

## 📝 License

MIT

---

Built by **[Ket Zero Consulting](https://ketzeroconsulting.ai)** 🚀
