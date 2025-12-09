# Architecture Overview

This document describes the technical architecture of the Marketing Measurement Hub.

## System Components

### 1. Frontend Layer (Streamlit)

The UI is built with Streamlit and consists of four main tabs:

| Tab | Purpose |
|-----|---------|
| **Agents** | Chat interface for interacting with AI agents |
| **Test Tracker** | View and manage all marketing tests |
| **Master Test Detail** | Deep-dive into individual test results |
| **Dataset Explorer** | Browse available datasets with visualizations |

Key UI features:
- Custom CSS theming (forest green `#16412F`, cream `#F5EBDD`)
- Responsive tab navigation
- Real-time streaming agent responses
- Session state management for multi-turn conversations

### 2. Agent Orchestration Layer (Google ADK)

We use Google's Agent Development Kit for multi-agent orchestration:

```
User Query
    │
    ▼
┌─────────────────┐
│   Lira Maven    │  ← Lead Agent (always receives first)
│  (Orchestrator) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────┐
│  Ketin  │ │  Brena  │
│  Vale   │ │ Colette │
└─────────┘ └─────────┘
Test Design  Measurement
```

**Agent Handoff Flow:**
1. All queries go to Lira first
2. Lira determines intent and routes to specialist
3. Specialists can hand back to Lira or directly to each other
4. Context is preserved across handoffs via session state

### 3. Tools Layer

Agents have access to shared tools defined in `agents/tools.py`:

| Tool | Description |
|------|-------------|
| `query_data` | Execute pandas queries on loaded datasets |
| `create_chart` | Generate Plotly visualizations |
| `search_knowledge_base` | RAG search over causal inference docs |
| `save_test_to_master` | Persist test configurations |
| `run_causal_impact` | Execute CausalImpact measurement |
| `validate_audience_split` | Check test/control balance |

### 4. Data Layer

**Local Development:**
- CSV files in `data/` directory
- ChromaDB vector store in `knowledge_base/chroma_db/`

**Production (Cloud Run):**
- Data stored in Google Cloud Storage (`gs://ol-measurement-hub-data/`)
- `gcs_loader.py` abstracts local vs cloud file access
- Caching via `@st.cache_data` with TTL

### 5. Causal Inference Engine

We use `tfp-causalimpact` for Bayesian structural time series:

```python
# Simplified flow
pre_period = [start_date, intervention_date]
post_period = [intervention_date, end_date]

model = CausalImpact(data, pre_period, post_period)
summary = model.summary()  # Point estimate, credible intervals
```

Key outputs:
- **Average lift**: Daily incremental effect
- **Cumulative lift**: Total incremental effect over test period
- **Credible intervals**: 95% Bayesian confidence bounds
- **P-value equivalent**: Frequentist interpretation

## Data Flow

```
1. User sends message
       │
       ▼
2. Streamlit captures input, adds to session state
       │
       ▼
3. ADK Runner processes message through agent graph
       │
       ▼
4. Agent calls tools (query_data, search_kb, etc.)
       │
       ▼
5. Tool results returned to agent
       │
       ▼
6. Agent generates response (streamed to UI)
       │
       ▼
7. Response + artifacts saved to session state
       │
       ▼
8. UI re-renders with new content
```

## Session Management

Each user session maintains:
- `chat_history`: List of user/agent message pairs
- `current_test_id`: Active test being discussed
- `agent_state`: ADK session state for context
- `loaded_data`: Cached dataframes

Sessions are ephemeral (lost on page refresh in this POC).

## Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                 Cloud Run                        │
│  ┌───────────────────────────────────────────┐  │
│  │         Streamlit Container               │  │
│  │  • app.py (main application)              │  │
│  │  • Gemini API calls (Vertex AI)           │  │
│  │  • GCS data access                        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  Vertex AI      │  │  Cloud Storage  │
│  (Gemini 2.5)   │  │  (Data + Files) │
└─────────────────┘  └─────────────────┘
         
┌─────────────────────────────────────────────────┐
│            Cloud Scheduler (Daily)              │
│  └──► Cloud Function (purge_test_data)          │
│       └──► Resets GCS data to baseline          │
└─────────────────────────────────────────────────┘
```

## Security Considerations

- **Authentication**: Cloud Run service account for GCS/Vertex AI
- **Secrets**: Environment variables (not in code)
- **Data**: Sample/synthetic data only (no PII)
- **Public access**: Demo is unauthenticated (POC only)
