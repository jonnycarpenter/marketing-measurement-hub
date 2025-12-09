"""
Outdoorsy Living Marketing Measurement Hub
A multi-agent Streamlit application for designing marketing tests and measuring impact.

Agents:
- Lira Maven (Measurement Lead) - Sandbox tab
- Ketin Vale (Test Design Agent) - Test Design tab  
- Brena Colette (Measurement Agent) - Advanced Statistics & Business Impact tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google ADK imports
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google import genai

# Configure Vertex AI for genai client
# This must happen before any ADK agents are created
project = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if project:
    # Initialize the default genai client with Vertex AI credentials
    genai_client = genai.Client(
        vertexai=True,
        project=project,
        location=location
    )
    print(f"Initialized genai client for Vertex AI: project={project}, location={location}")
else:
    print("WARNING: GOOGLE_CLOUD_PROJECT not set - Vertex AI client not initialized")
    
if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
    print("WARNING: GOOGLE_CLOUD_LOCATION not set - using default 'us-central1'")

# Register Claude model class for Vertex AI usage
# Required when using Claude model strings directly via Vertex AI with LlmAgent
try:
    from google.adk.models.anthropic_llm import Claude
    from google.adk.models.registry import LLMRegistry
    LLMRegistry.register(Claude)
    print("Claude model registered successfully for Vertex AI")
except ImportError as e:
    print(f"WARNING: Could not register Claude model - anthropic[vertex] may not be installed: {e}")

# Local imports
from agents.measurement_lead import measurement_lead_agent
from agents.test_design_agent import test_design_agent
from agents.measurement_agent import measurement_agent
from utils.data_loader import DataLoader
from utils.gcs_loader import gcs_loader
from utils.model_manager import get_model_manager
from utils.trace_collector import (
    initialize_otel_tracer,
    clear_trace_data,
    render_trace_sidebar,
    add_manual_trace
)
from agents.tools import session_context


# =============================================================================
# GCS HELPER FUNCTIONS (for seamless local/cloud file access)
# =============================================================================

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load CSV from GCS (production) or local (development)."""
    return gcs_loader.read_csv(path, **kwargs)


def file_exists(path: str) -> bool:
    """Check if file exists in GCS (production) or local (development)."""
    return gcs_loader.exists(path)


def read_file_bytes(path: str) -> bytes:
    """Read file bytes from GCS (production) or local (development)."""
    return gcs_loader.read_bytes(path)


def list_files(prefix: str, suffix: str = None) -> list:
    """List files with prefix from GCS (production) or local (development)."""
    return gcs_loader.list_files(prefix, suffix)


# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="Outdoorsy Living | Measurement Hub",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Outdoorsy Living Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ========================================
       OUTDOORSY LIVING DESIGN SYSTEM
       ======================================== */

    :root {
        /* Brand palette */
        --ol-forest: #16412F;        /* primary green */
        --ol-forest-dark: #0F261A;
        --ol-forest-soft: #2A5B3A;
        --ol-cream: #F5EBDD;         /* main background */
        --ol-offwhite: #FFF9F1;      /* cards / panels */
        --ol-charcoal: #2F3437;      /* main text */
        --ol-sage: #A7BFA5;          /* subtle highlight */
        --ol-rust: #C56B33;          /* accent / CTA */
        --ol-border: #E4D8C9;

        /* Legacy color aliases for compatibility */
        --ol-green: #16412F;
        --ol-green-light: #2A5B3A;
        --ol-green-dark: #0F261A;

        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.08);

        /* Spacing (8px base) */
        --space-xs: 0.5rem;
        --space-sm: 1rem;
        --space-md: 1.5rem;
        --space-lg: 2rem;
        --space-xl: 3rem;

        /* Animation */
        --transition-fast: 140ms ease-out;
        --transition-base: 220ms ease-out;
    }

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        -webkit-font-smoothing: antialiased;
        background-color: var(--ol-cream);
        color: var(--ol-charcoal);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        animation: fadeIn 0.4s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ========================================
       TYPOGRAPHY
       ======================================== */

    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--ol-forest-dark);
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }

    .sub-header {
        font-size: 1.05rem;
        color: rgba(47, 52, 55, 0.78);
        margin-bottom: var(--space-lg);
        max-width: 46rem;
        font-weight: 400;
        line-height: 1.6;
        letter-spacing: 0.2px;
    }
    
    h3 {
        font-weight: 600;
        color: var(--ol-forest-dark);
        letter-spacing: -0.02em;
    }

    /* Small label style (for section tags) */
    .ol-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: rgba(47, 52, 55, 0.7);
    }

    /* ========================================
       CARDS & CONTAINERS
       ======================================== */

    .ol-card {
        background: var(--ol-offwhite);
        padding: var(--space-md);
        border-radius: 14px;
        border: 1px solid var(--ol-border);
        box-shadow: var(--shadow-sm);
        margin-bottom: var(--space-sm);
        transition: box-shadow var(--transition-base), transform var(--transition-base), border-color var(--transition-base);
    }

    .ol-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
        border-color: var(--ol-forest-soft);
    }

    .info-card {
        background: linear-gradient(145deg, #FFFDF8 0%, var(--ol-offwhite) 40%, #F2E5D3 100%);
        border-radius: 14px;
        padding: var(--space-md);
        border: 1px solid rgba(0, 0, 0, 0.04);
        box-shadow: var(--shadow-sm);
        margin-bottom: var(--space-sm);
    }

    /* ========================================
       METRIC CARDS
       ======================================== */

    .metric-card {
        background: var(--ol-forest);
        padding: var(--space-md);
        border-radius: 14px;
        color: #FFFFFF;
        text-align: left;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top left, rgba(255,255,255,0.16), transparent 55%);
        pointer-events: none;
    }

    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.15rem;
    }

    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        opacity: 0.9;
    }

    /* Metric card header/value (Summary Impact tab) */
    .metric-card-header {
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .metric-card-delta {
        font-size: 0.8rem;
        font-weight: 500;
        opacity: 0.85;
    }
    
    .delta-positive {
        color: #c8e6c9;
    }
    
    .delta-negative {
        color: #ffcdd2;
    }
    
    .delta-neutral {
        opacity: 0.75;
    }
    
    /* Results highlight boxes */
    .results-highlight {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid var(--ol-green);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .results-highlight h3 {
        margin: 0 0 0.5rem 0;
        color: #2e7d32;
        font-size: 1.1rem;
    }
    
    .results-highlight p {
        margin: 0;
        color: #1b5e20;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .results-highlight-negative {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: #c62828;
    }
    
    .results-highlight-negative h3 {
        color: #c62828;
    }
    
    .results-highlight-negative p {
        color: #b71c1c;
    }
    
    .results-highlight-caution {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
    }
    
    .results-highlight-caution h3 {
        color: #b45309;
    }
    
    .results-highlight-caution p {
        color: #92400e;
    }

    /* ========================================
       CHAT STYLING
       ======================================== */

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-sm);
        padding: var(--space-sm);
        max-height: 70vh;
        overflow-y: auto;
    }

    .chat-bubble {
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        max-width: 85%;
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .chat-user {
        background: #e3e8dc;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    .chat-bot {
        background: var(--ol-offwhite);
        align-self: flex-start;
        margin-right: auto;
        border-left: 3px solid var(--ol-forest);
        border-bottom-left-radius: 4px;
    }

    /* ========================================
       BUTTONS & INTERACTIVE ELEMENTS
       ======================================== */

    .stButton > button {
        border-radius: 24px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.6rem 1.6rem;
        transition: background var(--transition-base), box-shadow var(--transition-base);
        box-shadow: var(--shadow-sm);
        border: none;
    }

    .stButton > button:hover {
        box-shadow: var(--shadow-md);
    }

    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: var(--ol-forest);
        color: #FFFFFF;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--ol-forest-dark);
    }

    /* ========================================
       SIDEBAR
       ======================================== */

    [data-testid="stSidebar"] {
        background: var(--ol-cream);
        border-right: 1px solid var(--ol-border);
    }

    [data-testid="stSidebar"] .css-1d391kg {
        background-color: transparent;
    }

    /* ========================================
       STATUS BADGES
       ======================================== */

    .badge-success, .badge-warning, .badge-info {
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        display: inline-block;
    }

    .badge-success {
        background: #d4edda;
        color: #155724;
    }

    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }

    .badge-info {
        background: #d1ecf1;
        color: #0c5460;
    }

    /* ========================================
       CHECKLIST ITEMS
       ======================================== */

    .checklist-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        display: flex;
        align-items: center;
        background: var(--ol-offwhite);
        border: 1px solid var(--ol-border);
        transition: border-color var(--transition-base);
    }

    .checklist-item:hover {
        border-color: var(--ol-forest-soft);
    }

    .checklist-done {
        border-left: 3px solid var(--ol-success);
    }

    .checklist-pending {
        border-left: 3px solid #BDBDBD;
        opacity: 0.75;
    }

    /* ========================================
       TABS (Pill Style)
       ======================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
        padding: 0;
        border-bottom: 1px solid var(--ol-border);
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.5rem 1.2rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        transition: background var(--transition-base), color var(--transition-base);
        background: transparent;
        color: var(--ol-charcoal);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(22, 65, 47, 0.08);
    }

    .stTabs [aria-selected="true"] {
        background: var(--ol-forest) !important;
        color: #FFFFFF !important;
    }

    /* ========================================
       WORKSPACE TAB SELECTOR (Radio styled as Tabs)
       ======================================== */
    
    /* Container for horizontal radio */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stRadio"]) {
        margin-bottom: 0 !important;
    }
    
    /* Hide the radio group label */
    div[data-testid="stRadio"] > label:first-child {
        display: none !important;
    }
    
    /* Radio container as tab bar */
    div[data-testid="stRadio"] > div {
        gap: 6px !important;
        background: transparent;
        padding: 0;
        flex-wrap: wrap;
    }
    
    /* Individual radio options as tabs */
    div[data-testid="stRadio"] > div > label {
        padding: 0.35rem 0.7rem !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.01em;
        transition: background var(--transition-base), color var(--transition-base);
        background: transparent !important;
        color: var(--ol-charcoal) !important;
        border: 1px solid var(--ol-border) !important;
        cursor: pointer;
        margin: 0 !important;
    }
    
    /* Hide the actual radio circle */
    div[data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }
    
    /* Hover state */
    div[data-testid="stRadio"] > div > label:hover {
        background: rgba(22, 65, 47, 0.08) !important;
        border-color: var(--ol-forest-soft) !important;
    }
    
    /* Selected state - dark green background */
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background: var(--ol-forest) !important;
        color: #FFFFFF !important;
        border-color: var(--ol-forest) !important;
    }

    /* ========================================
       USER AVATAR STYLING
       ======================================== */
    
    /* Style user chat avatar as forest green circle with 'Me' */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageAvatarUser"] {
        background: var(--ol-forest) !important;
        border-radius: 50% !important;
        width: 2rem !important;
        height: 2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    div[data-testid="stChatMessageAvatarUser"]::before {
        content: 'Me' !important;
        color: white !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    
    /* Hide the default user icon SVG */
    div[data-testid="stChatMessageAvatarUser"] svg {
        display: none !important;
    }

    /* ========================================
       GO/NO GO BADGES
       ======================================== */

    .go-badge, .no-go-badge {
        font-size: 2.2rem;
        font-weight: 800;
        padding: 1.4rem 2.8rem;
        border-radius: 14px;
        display: inline-block;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .go-badge {
        background: var(--ol-forest);
        color: #FFFFFF;
        box-shadow: var(--shadow-lg);
    }

    .no-go-badge {
        background: #B91C1C;
        color: #FFFFFF;
        box-shadow: var(--shadow-lg);
    }

    /* ========================================
       UTILITY CLASSES
       ======================================== */

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .loading-skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
        border-radius: 8px;
    }

    /* Smooth Fade */
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    
    # Lira's welcome message on initial load
    lira_welcome_message = """Welcome to the proof-of-concept **Marketing Measurement Hub (MMH)**.

**Lira** – your MMH guide will be able to answer any general* questions you have and give you a tour of the facilities – **Just say "Hi Lira" in this chat** and she'll jump right in 😊

**UX Shortcomings:**
- This POC is built on Streamlit, an app UI meant for fast prototyping. I pressed it pretty close to the limit, so please don't judge based on UI/UX hiccups (you will almost certainly experience some).

**Pro Tips:**
- The agents will remind you, but you have to manually navigate to different Labs to see what the agent is doing.
- Agent handoffs – The handoffs between agents are a little dicey; they don't always pass you along when they should or pass you along too early.
  - When you're ready to design a test, say something like "let me talk to Ketin" or "let's design a test"
  - Same thing for Brena, the measurement specialist

---

If you want to connect to provide feedback, explore hooking up your own data to see it in action in your context, or talk through another AI-powered workflow we could potentially collaborate on – shoot me a [LinkedIn DM](https://www.linkedin.com/in/jonny-carpenter-86070110/) or send a note to **jonny@ketzeroconsulting.ai**"""
    
    defaults = {
        'chat_history': [{
            'role': 'assistant',
            'content': lira_welcome_message,
            'agent': 'lira',
            'timestamp': datetime.now().isoformat()
        }],  # Each message: {role, content, agent, timestamp}
        'current_tab': 'sandbox',
        'current_agent': 'lira',  # Currently active agent (updates on transfer)
        'last_responding_agent': 'lira',  # Track which agent last responded
        'test_design': {
            'test_name': None,
            'test_description': None,
            'start_date': None,
            'end_date': None,
            'split_type': None,
            'test_regions': [],
            'control_regions': [],
            'checklist': {}
        },
        'analysis_results': None,
        'adk_session_id': f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'runners': {},
        'show_files_modal': False,
        'selected_file_view': None,
        'agent_processing': False,
        'shared_agent_context': {
            'objective': None,
            'current_phase': 'planning',
            'key_decisions': [],
            'notes': [],
            'generated_artifacts': []
        },
        'show_open_me_dialog': False,  # For the OPEN ME intro modal
        # Track last designed test for Brena context
        'last_designed_test_id': None,
        'last_designed_test_name': None,
        # Dataset Explorer - completely isolated from agent workflow
        'explorer_chat_history': [],
        'explorer_results': None,  # Stores latest query results (df, chart, etc.)
        'explorer_processing': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# BACKGROUND INITIALIZATION
# =============================================================================

@st.cache_resource
def init_rag_system():
    """
    Initialize the RAG system (KnowledgeBaseManager) in the background.
    This prevents the first search from hanging the UI.
    """
    try:
        from utils.rag_utils import get_kb_manager
        # This will trigger the singleton initialization
        manager = get_kb_manager()
        return manager
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return None

# Trigger initialization
init_rag_system()


# =============================================================================
# DATA LOADING
# =============================================================================

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def get_data_loader():
    """
    Get the singleton DataLoader instance.
    Cached resource ensures we don't recreate the loader (and potentially re-read configs).
    """
    return DataLoader(data_dir="data")

@st.cache_data
def load_data():
    """Load all data files."""
    loader = get_data_loader()
    try:
        return {
            'sales': loader.load_sales_data(),
            'customers': loader.load_customer_file(),
            'products': loader.load_product_inventory(),
            'tests': loader.load_master_testing_doc(),
            'promos': loader.load_promo_calendar()
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}


@st.cache_data
def get_quick_stats():
    """Calculate quick stats for display."""
    loader = get_data_loader()
    try:
        sales = loader.load_sales_data()
        customers = loader.load_customer_file()
        
        sales['order_date'] = pd.to_datetime(sales['order_date'])
        
        return {
            'total_orders': len(sales),
            'total_revenue': sales['order_value_usd'].sum(),
            'total_customers': len(customers),
            'avg_order_value': sales['order_value_usd'].mean(),
            'dma_count': sales['dma'].nunique(),
            'date_range': f"{sales['order_date'].min().strftime('%b %Y')} - {sales['order_date'].max().strftime('%b %Y')}"
        }
    except Exception as e:
        return {}


# =============================================================================
# AGENT RUNNERS
# =============================================================================

def get_agent_runner(agent_name: str, force_refresh: bool = False):
    """Get or create an ADK runner for the specified agent.
    
    Args:
        agent_name: The name of the agent ('lira', 'ketin', 'brena')
        force_refresh: If True, recreate the agent with current model (for fallback)
    """
    # Import factory functions for dynamic agent creation
    from agents.measurement_lead import create_measurement_lead_agent
    from agents.test_design_agent import create_test_design_agent  
    from agents.measurement_agent import create_measurement_agent
    
    if force_refresh and agent_name in st.session_state.runners:
        del st.session_state.runners[agent_name]
        
    if agent_name not in st.session_state.runners:
        try:
            session_service = InMemorySessionService()
            model_manager = get_model_manager()
            current_model = model_manager.get_current_model(agent_name)
            
            # Create agent with current dynamic model
            if agent_name == 'lira':
                agent = create_measurement_lead_agent(current_model)
            elif agent_name == 'ketin':
                agent = create_test_design_agent(current_model)
            elif agent_name == 'brena':
                agent = create_measurement_agent(current_model)
            else:
                agent = create_measurement_lead_agent(current_model)
            
            # Log which model is being used
            model_info = model_manager.get_model_display_info(agent_name)
            if model_info['fallback_triggered']:
                print(f"[{agent_name}] Using fallback model: {current_model}")
            else:
                print(f"[{agent_name}] Using primary model: {current_model}")
            
            # Create runner without pre-creating session - let it create on first use
            runner = Runner(
                agent=agent,
                app_name="measurement_hub",
                session_service=session_service
            )
            
            st.session_state.runners[agent_name] = runner
            
        except Exception as e:
            st.error(f"Agent setup issue: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    return st.session_state.runners.get(agent_name)


async def run_agent_async(agent_name: str, query: str) -> str:
    """Run a query through the specified agent."""
    runner = get_agent_runner(agent_name)
    if runner is None:
        return "Agent not available. Please check your GOOGLE_API_KEY."
    
    try:
        # Set processing state for Matrix visualization
        st.session_state.agent_processing = True
        clear_trace_data()
        add_manual_trace(f"⟩ Initializing {agent_name.title()} Agent", "OK", 0)
        add_manual_trace(f"⟩ Parsing Query Input", "OK", 5)
        add_manual_trace(f"⟩ Activating Neural Networks", "OK", 10)
        
        # Set session context for tools to ensure thread safety
        token = session_context.set(st.session_state.adk_session_id)
        
        try:
            # Ensure session exists - create if not already created for this agent
            session_key = f"session_created_{agent_name}"
            if session_key not in st.session_state:
                await runner.session_service.create_session(
                    app_name="measurement_hub",
                    user_id="streamlit_user",
                    session_id=st.session_state.adk_session_id
                )
                st.session_state[session_key] = True
            
            # Inject recent test context for Brena when user gives affirmative response
            enhanced_query = query
            if agent_name == 'brena' and st.session_state.get('last_designed_test_id'):
                affirmative_words = ['yes', 'yeah', 'sure', 'go ahead', 'please', 'measure it', 'run it', 'do it', 'yep', 'ok', 'okay']
                if any(word in query.lower().strip() for word in affirmative_words) or query.lower().strip() in affirmative_words:
                    test_id = st.session_state['last_designed_test_id']
                    test_name = st.session_state.get('last_designed_test_name', 'the test')
                    enhanced_query = f"[CONTEXT: User confirmed they want to measure the recently designed test: {test_id} ({test_name}). Proceed with run_causal_impact_analysis for test_id='{test_id}']\n\nUser said: {query}"
            
            content = types.Content(
                role="user",
                parts=[types.Part(text=enhanced_query)]
            )
            
            add_manual_trace(f"⟩ Connecting to LLM", "OK", 20)
            add_manual_trace(f"⟩ Context Window Loaded", "OK", 35)
            
            response_text = ""
            chart_outputs = []  # Capture chart tool outputs
            event_count = 0
            async for event in runner.run_async(
                user_id="streamlit_user",
                session_id=st.session_state.adk_session_id,
                new_message=content
            ):
                event_count += 1
                if event_count == 1:
                    add_manual_trace(f"⟩ Processing Stream", "OK", 50)
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text is not None:
                                response_text += part.text
                            # Capture function responses that contain image data
                            if hasattr(part, 'function_response') and part.function_response:
                                try:
                                    func_response = part.function_response
                                    func_name = getattr(func_response, 'name', 'unknown')
                                    print(f"[CHART-DEBUG] Function response from: {func_name}")
                                    if hasattr(func_response, 'response'):
                                        resp_data = func_response.response
                                        print(f"[CHART-DEBUG] Response type: {type(resp_data)}, keys: {resp_data.keys() if isinstance(resp_data, dict) else 'N/A'}")
                                        
                                        # ADK wraps function responses in {'result': actual_data}
                                        # Unwrap if present
                                        if isinstance(resp_data, dict) and 'result' in resp_data and len(resp_data) == 1:
                                            resp_data = resp_data['result']
                                            print(f"[CHART-DEBUG] Unwrapped 'result', new type: {type(resp_data)}")
                                        
                                        # Parse string responses to dict
                                        if isinstance(resp_data, str):
                                            try:
                                                resp_data = json.loads(resp_data)
                                                print(f"[CHART-DEBUG] Parsed JSON keys: {resp_data.keys() if isinstance(resp_data, dict) else 'N/A'}")
                                            except json.JSONDecodeError:
                                                print(f"[CHART-DEBUG] Failed to parse JSON response")
                                                pass
                                        
                                        if isinstance(resp_data, dict):
                                            # Check direct image_base64
                                            if 'image_base64' in resp_data:
                                                print(f"[CHART-DEBUG] Found direct image_base64!")
                                                chart_outputs.append(json.dumps(resp_data))
                                            # Check nested pre_period_chart (from save_current_test_design)
                                            elif 'pre_period_chart' in resp_data:
                                                print(f"[CHART-DEBUG] Found pre_period_chart!")
                                                nested_chart = resp_data['pre_period_chart']
                                                if isinstance(nested_chart, dict) and 'charts' in nested_chart:
                                                    for chart in nested_chart['charts']:
                                                        if 'image_base64' in chart:
                                                            print(f"[CHART-DEBUG] Extracted chart: {chart.get('type', 'unknown')}")
                                                            chart_outputs.append(json.dumps(chart))
                                            # Check charts array directly
                                            elif 'charts' in resp_data:
                                                print(f"[CHART-DEBUG] Found charts array with {len(resp_data['charts'])} charts")
                                                for chart in resp_data['charts']:
                                                    if isinstance(chart, dict) and 'image_base64' in chart:
                                                        chart_outputs.append(json.dumps(chart))
                                            else:
                                                print(f"[CHART-DEBUG] No chart data found in keys: {list(resp_data.keys())[:10]}")
                                except Exception as e:
                                    print(f"[CHART-DEBUG] Error processing function response: {e}")
                                    pass  # Silently skip unparseable function responses
            
            add_manual_trace(f"⟩ Received {event_count} Events", "OK", 75)
            
            # If we captured chart outputs, append them to the response for rendering
            if chart_outputs:
                for chart_json in chart_outputs:
                    response_text += f"\n\n```json\n{chart_json}\n```"
            
            add_manual_trace(f"⟩ Response Synthesized", "OK", 100)
            add_manual_trace(f"⟩ Output Encoded", "OK", 15)
            st.session_state.agent_processing = False
            return response_text if response_text else "Processing complete."
            
        finally:
            # Reset context to avoid leaking into other requests
            session_context.reset(token)
    
    except Exception as e:
        import traceback
        error_str = str(e)
        # Log full error to console for debugging
        print(f"Agent error: {error_str}")
        print(traceback.format_exc())
        
        # Check if this is a token limit error and trigger model fallback
        model_manager = get_model_manager()
        if model_manager.should_fallback(e):
            if model_manager.trigger_fallback(agent_name, e):
                # Clear the runner cache to force recreation with new model
                if agent_name in st.session_state.runners:
                    del st.session_state.runners[agent_name]
                # Clear session to force recreation
                session_key = f"session_created_{agent_name}"
                if session_key in st.session_state:
                    del st.session_state[session_key]
                
                new_model = model_manager.get_current_model(agent_name)
                add_manual_trace(f"⟩ Switching to {new_model}", "RETRY", 0)
                print(f"[Model Fallback] Switching {agent_name} to {new_model} due to token limit")
                
                # Retry with the new model
                st.session_state.agent_processing = False
                return await run_agent_async(agent_name, query)
        
        add_manual_trace(f"✗ Error: {error_str[:30]}", "ERROR", 0)
        st.session_state.agent_processing = False
        
        # Provide friendlier error messages for common issues
        if "'NoneType' object is not iterable" in error_str:
            return "I encountered a data processing issue. The operation may have partially completed. Please try again or check if your request was processed."
        elif "'NoneType' object has no attribute" in error_str:
            return "I had trouble accessing some data. This might be a temporary issue - please try again."
        else:
            return f"Error: {error_str}"


def run_agent(agent_name: str, query: str) -> str:
    """Synchronous wrapper for agent execution."""
    try:
        # Use asyncio.run which handles event loop creation properly
        return asyncio.run(run_agent_async(agent_name, query))
    except RuntimeError as e:
        # If there's already an event loop running, use nest_asyncio approach
        if "cannot be called from a running event loop" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(run_agent_async(agent_name, query))
        return f"Error running agent: {e}"
    except Exception as e:
        return f"Error running agent: {e}"


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def get_image_base64(image_path: str) -> Optional[str]:
    """Load image and return base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


def render_chat_response(response: str):
    """
    Render a chat response, handling embedded charts and images.
    Looks for base64 image data in JSON responses and displays them inline.
    Also handles HTML img tags with base64 data.
    Also handles markdown image links to local files.
    """
    import re
    from pathlib import Path
    
    # Check for markdown image links with local file paths
    # Pattern: ![alt text](path/to/file.png)
    md_img_pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|webp))\)'
    md_img_matches = re.findall(md_img_pattern, response, re.IGNORECASE)
    
    if md_img_matches:
        displayed_any = False
        clean_response = response
        
        for alt_text, img_path in md_img_matches:
            try:
                # Check if it's a local file path
                path = Path(img_path)
                if path.exists() and path.is_file():
                    # Read and display the image
                    st.image(str(path), caption=alt_text if alt_text else None)
                    displayed_any = True
                    # Remove the markdown image from response
                    clean_response = clean_response.replace(f'![{alt_text}]({img_path})', '')
            except Exception as e:
                print(f"Warning: Failed to display local image {img_path}: {e}")
        
        if displayed_any:
            clean_response = clean_response.strip()
            if clean_response:
                st.markdown(clean_response)
            return
    
    # Check for HTML img tags with base64 data
    img_pattern = r'<img\s+src="data:image/[^;]+;base64,([^"]+)"[^>]*>'
    img_matches = re.findall(img_pattern, response)
    
    if img_matches:
        # Extract and display all images
        for img_data in img_matches:
            try:
                st.image(f"data:image/png;base64,{img_data}")
            except Exception as e:
                st.error(f"Failed to display image: {e}")
        
        # Remove img tags from response and display remaining text
        clean_response = re.sub(img_pattern, '', response).strip()
        if clean_response:
            st.markdown(clean_response)
        return
    
    # Check if response contains chart JSON with image_base64
    try:
        # Look for JSON blocks that might contain chart data
        json_pattern = r'\{[^{}]*\"image_base64\"[^{}]*\}'
        
        # Also check if the entire response is JSON
        if response.strip().startswith('{'):
            try:
                data = json.loads(response)
                if isinstance(data, dict) and 'image_base64' in data:
                    # Display the chart
                    image_data = data['image_base64']
                    st.image(f"data:image/png;base64,{image_data}")
                    
                    # Display summary if available
                    if 'summary' in data:
                        summary = data['summary']
                        st.caption(f"📊 **Summary**: {summary.get('groups_shown', 'N/A')} groups | "
                                  f"Total: {summary.get('metric_total', 0):,.0f} | "
                                  f"Top: {summary.get('top_group', 'N/A')}")
                    return
            except json.JSONDecodeError:
                pass
        
        # Check for embedded JSON with chart data
        json_matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        displayed_chart = False
        clean_response = response
        
        for match in json_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and 'image_base64' in data:
                    # Display the chart
                    image_data = data['image_base64']
                    st.image(f"data:image/png;base64,{image_data}")
                    displayed_chart = True
                    
                    # Display summary if available
                    if 'summary' in data:
                        summary = data['summary']
                        st.caption(f"📊 **Summary**: {summary.get('groups_shown', 'N/A')} groups | "
                                  f"Total: {summary.get('metric_total', 0):,.0f} | "
                                  f"Top: {summary.get('top_group', 'N/A')}")
                    
                    # Remove the JSON block from the response
                    clean_response = clean_response.replace(f'```json\n{match}\n```', '')
                    clean_response = clean_response.replace(f'```json{match}```', '')
            except json.JSONDecodeError:
                continue
        
        # Display remaining text
        clean_response = clean_response.strip()
        if clean_response:
            st.markdown(clean_response)
        elif not displayed_chart:
            st.markdown(response)
            
    except Exception as e:
        # Fallback to plain text display
        st.markdown(response)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT_AVATARS = {
    'lira': 'agent_persona_injection_andOL_logo/Lira_avatar.png',
    'ketin': 'agent_persona_injection_andOL_logo/Ketin_avatar.png',
    'brena': 'agent_persona_injection_andOL_logo/Brena_avatar.png'
}

AGENT_TITLES = {
    'lira': 'Lira Maven',
    'ketin': 'Ketin Vale',
    'brena': 'Brena Colette'
}

AGENT_ROLES = {
    'lira': 'Measurement Lead',
    'ketin': 'Test Design Specialist',
    'brena': 'Measurement Analyst'
}


def get_agent_avatar_path(agent_name: str) -> Optional[str]:
    """Get the avatar path for an agent, returns None if not found."""
    path = AGENT_AVATARS.get(agent_name)
    if path and os.path.exists(path):
        return path
    return None


def render_agent_avatar(agent_name: str):
    """Render the agent avatar in sidebar."""
    path = get_agent_avatar_path(agent_name)
    if path:
        st.sidebar.image(path, width=120)
    
    st.sidebar.markdown(f"**{AGENT_TITLES.get(agent_name, 'Agent')}**")
    st.sidebar.caption(AGENT_ROLES.get(agent_name, ''))


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with logo, agent avatar, and tracing."""
    
    # "Under the Hood" link at the very top - links to Figma workflow
    st.sidebar.markdown('<div style="margin-top: -0.5rem;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<a href="https://ketzeroconsulting.ai/mmh-under-the-hood.html" target="_blank" '
        'style="display: block; text-align: center; padding: 0.5rem 1rem; '
        'background: linear-gradient(135deg, #16412F 0%, #1a5038 100%); '
        'color: white; text-decoration: none; border-radius: 8px; '
        'font-weight: 600; font-size: 0.85rem; margin-bottom: 0.5rem; '
        'transition: all 0.2s ease;">'
        '🔍 See How It Works</a>',
        unsafe_allow_html=True
    )
    
    # OPEN ME button
    # Create the OPEN ME button with dialog
    if st.sidebar.button("📖 OPEN ME", key="open_me_btn", use_container_width=True):
        st.session_state.show_open_me_dialog = True
    
    # Style the button with yellow background
    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] button[kind="secondary"]:first-of-type {
        background-color: #FFD700 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: 2px solid #000000 !important;
        padding: 0.6rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:first-of-type:hover {
        background-color: #FFC000 !important;
        border-color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Logo - Use the new Outdoorsy Living logo with reduced top padding
    logo_path = 'agent_persona_injection_andOL_logo/outdoorsy_living_logo_new.png'
    
    # Add negative margin to move logo up
    st.sidebar.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
    else:
        # Fallback to old path or text
        alt_logo_path = 'streamlit_reimagining/outdoorsy_living_NEW_logo.png'
        if os.path.exists(alt_logo_path):
            st.sidebar.image(alt_logo_path, width=200)
        else:
            st.sidebar.markdown("## 🏔️ Outdoorsy Living")
    
    # Marketing Measurement Hub title under logo
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: -0.5rem; margin-bottom: 0.5rem;">
        <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.12em; color: #16412F; line-height: 1.3;">
            MARKETING<br>MEASUREMENT HUB
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Current Agent
    st.sidebar.markdown("### 💬 Your Agent")
    render_agent_avatar(st.session_state.current_agent)
    
    # Show current model info
    model_manager = get_model_manager()
    current_agent = st.session_state.current_agent
    model_info = model_manager.get_model_display_info(current_agent)
    model_name = model_info.get('display_name', model_info['current_model'])
    if model_info['fallback_triggered']:
        st.sidebar.caption(f"🔄 Model: {model_name} (fallback)")
    else:
        st.sidebar.caption(f"🤖 Model: {model_name}")
    
    # Tracing
    render_trace_sidebar()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Google ADK")
    
    # Ket Zero branding with logo inline - using HTML/markdown for tight spacing
    ketzero_logo_path = 'agent_persona_injection_andOL_logo/KetZero_Logo_FINAL.png'
    if os.path.exists(ketzero_logo_path):
        # Encode logo as base64 for inline HTML
        with open(ketzero_logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        # Display text and logo inline with word-like spacing - clickable link to site
        st.sidebar.markdown(
            f'<a href="https://ketzeroconsulting.ai" target="_blank" style="text-decoration: none; color: inherit;">'
            f'<p style="font-size: 0.8rem; color: rgba(49, 51, 63, 0.6); margin: 0; display: flex; align-items: center; gap: 0.3em;">'
            f'Built by Ket Zero <img src="data:image/png;base64,{logo_data}" style="height: 1.2em; vertical-align: middle;">'
            f'</p></a>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown("[Built by Ket Zero](https://ketzeroconsulting.ai)")


def render_open_me_dialog():
    """Render the OPEN ME intro dialog/modal."""
    if st.session_state.get('show_open_me_dialog', False):
        # Reset the flag immediately so dialog doesn't reopen on next rerun
        # The dialog will still show for this render cycle
        st.session_state.show_open_me_dialog = False
        
        # Load the intro text
        intro_text = ""
        intro_file_path = "mktg_meas_hub_open_me_text_intro.txt"
        
        # Try to load from GCS first, then local
        try:
            intro_text = gcs_loader.read_text(intro_file_path)
        except:
            # Fallback to local file
            if os.path.exists(intro_file_path):
                with open(intro_file_path, 'r') as f:
                    intro_text = f.read()
            else:
                intro_text = "Welcome to the Marketing Measurement Hub! This is a proof of concept showcasing AI-powered marketing measurement capabilities."
        
        # Create modal using st.dialog (Streamlit 1.33+)
        @st.dialog("📖 Before You Dive In...", width="large")
        def show_intro_modal():
            st.markdown("""
            <style>
            .intro-content {
                font-family: 'Courier New', monospace;
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #FFD700;
                white-space: pre-wrap;
                font-size: 0.9rem;
                line-height: 1.6;
                max-height: 60vh;
                overflow-y: auto;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="intro-content">{intro_text}</div>', unsafe_allow_html=True)
            
            st.markdown("")
            if st.button("Got it, let's go! 🚀", use_container_width=True, type="primary"):
                st.rerun()
        
        show_intro_modal()


# =============================================================================
# PERSISTENT CHAT (Always-On)
# =============================================================================

def extract_test_id_from_response(response: str, user_prompt: str) -> Optional[str]:
    """
    Extract test ID from Brena's response or user prompt.
    Used to update the Advanced Statistics workspace display.
    
    Looks for:
    1. TEST-XXXXXXXX pattern in response or prompt
    2. Test name matches in master_testing_doc.csv
    """
    import re
    
    # Pattern for test IDs like TEST-12021539
    test_id_pattern = r'TEST-\d{8}'
    
    # First check the response for explicit test ID
    match = re.search(test_id_pattern, response)
    if match:
        return match.group(0)
    
    # Then check the user prompt for test ID
    match = re.search(test_id_pattern, user_prompt)
    if match:
        return match.group(0)
    
    # Try to match by test name from user prompt
    try:
        tests_df = load_csv('data/master_testing_doc.csv')
        user_lower = user_prompt.lower()
        
        for _, row in tests_df.iterrows():
            test_name = str(row.get('test_name', '')).lower()
            # Check if significant words from test name appear in user prompt
            if test_name and len(test_name) > 5:
                # Match if most of the test name words appear
                name_words = [w for w in test_name.split() if len(w) > 3]
                if name_words:
                    matches = sum(1 for w in name_words if w in user_lower)
                    if matches >= len(name_words) * 0.6:  # 60% word match
                        return row['test_id']
    except:
        pass
    
    return None


def detect_agent_transfer(response: str, current_agent: str) -> str:
    """
    Detect if the response indicates a transfer to another agent.
    Returns the new agent name if transfer detected, otherwise current agent.
    
    Detects both:
    1. Transfer phrases ("let me connect you with Ketin")
    2. Agent self-introductions ("I'm Ketin", "Hi, I'm Ketin Vale")
    """
    response_lower = response.lower()
    
    # Check for Ketin (transfer or self-introduction)
    if current_agent != 'ketin' and any(phrase in response_lower for phrase in [
        "transfer you to ketin", "handing off to ketin", "ketin will", 
        "ketin can help", "let me connect you with ketin", "ketin vale",
        "i'm ketin", "i am ketin", "my name is ketin", "this is ketin",
        "ketin here", "hey, i'm ketin", "hi, i'm ketin"
    ]):
        return 'ketin'
    
    # Check for Brena (transfer or self-introduction)
    if current_agent != 'brena' and any(phrase in response_lower for phrase in [
        "transfer you to brena", "handing off to brena", "brena will",
        "brena can help", "let me connect you with brena", "brena colette",
        "i'm brena", "i am brena", "my name is brena", "this is brena",
        "brena here", "hey, i'm brena", "hi, i'm brena"
    ]):
        return 'brena'
    
    # Check for Lira (transfer or self-introduction)
    if current_agent != 'lira' and any(phrase in response_lower for phrase in [
        "transfer you to lira", "handing off to lira", "lira will",
        "back to lira", "let me connect you with lira", "lira maven",
        "i'm lira", "i am lira", "my name is lira", "this is lira",
        "lira here", "hey, i'm lira", "hi, i'm lira"
    ]):
        return 'lira'
    
    return current_agent


def detect_user_agent_request(user_message: str, current_agent: str) -> Optional[str]:
    """
    Detect if the user is asking to speak with a specific agent.
    Returns the requested agent name if detected, otherwise None.
    """
    message_lower = user_message.lower()
    
    # Check if user wants to talk to Ketin
    if current_agent != 'ketin' and any(phrase in message_lower for phrase in [
        "talk to ketin", "speak to ketin", "speak with ketin", "can i talk to ketin",
        "let me talk to ketin", "connect me with ketin", "transfer to ketin",
        "i want ketin", "need ketin", "get ketin"
    ]):
        return 'ketin'
    
    # Check if user wants to talk to Brena
    if current_agent != 'brena' and any(phrase in message_lower for phrase in [
        "talk to brena", "speak to brena", "speak with brena", "can i talk to brena",
        "let me talk to brena", "connect me with brena", "transfer to brena",
        "i want brena", "need brena", "get brena"
    ]):
        return 'brena'
    
    # Check if user wants to talk to Lira
    if current_agent != 'lira' and any(phrase in message_lower for phrase in [
        "talk to lira", "speak to lira", "speak with lira", "can i talk to lira",
        "let me talk to lira", "connect me with lira", "transfer to lira",
        "i want lira", "need lira", "get lira", "back to lira"
    ]):
        return 'lira'
    
    return None


def render_persistent_chat():
    """
    Render the always-on chat interface that persists across all tabs.
    Returns a container for tab-specific content.
    """
    # Create two columns: chat (left) and workspace (right)
    col_chat, col_workspace = st.columns([4, 6], gap="large")
    
    with col_chat:
        # Chat header - collaboration theme
        current_agent = st.session_state.current_agent
        st.markdown("### 🤝 Agent Collaboration")
        
        # Chat container with scrollable history
        chat_container = st.container(height=550)
        
        # Render all chat history (unified across tabs)
        with chat_container:
            for msg in st.session_state.chat_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                msg_agent = msg.get('agent', 'lira')
                
                if role == 'assistant':
                    # Get avatar for the agent who sent this message
                    avatar_path = get_agent_avatar_path(msg_agent)
                    with st.chat_message("assistant", avatar=avatar_path):
                        render_chat_response(content)
                else:
                    with st.chat_message("user"):
                        st.write(content)
        
        # Chat input - always available
        if prompt := st.chat_input("Message the team...", key="persistent_chat_input"):
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': prompt,
                'agent': None,  # User messages don't have an agent
                'timestamp': datetime.now().isoformat()
            })
            
            # Track if transfer happens
            agent_transferred = False
            current_agent = st.session_state.current_agent
            
            # Check if user is explicitly requesting a different agent
            requested_agent = detect_user_agent_request(prompt, current_agent)
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)
                
                if requested_agent:
                    # User explicitly asked for another agent - do direct handoff
                    old_agent_name = AGENT_TITLES.get(current_agent, 'Agent')
                    new_agent_name = AGENT_TITLES.get(requested_agent, 'Agent')
                    
                    # Current agent says goodbye
                    avatar_path = get_agent_avatar_path(current_agent)
                    handoff_msg = f"Of course! Let me connect you with {new_agent_name}. 🤝"
                    
                    with st.chat_message("assistant", avatar=avatar_path):
                        st.write(handoff_msg)
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': handoff_msg,
                        'agent': current_agent,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # New agent acknowledges - with context if Brena and recent test exists
                    new_avatar_path = get_agent_avatar_path(requested_agent)
                    
                    if requested_agent == 'brena' and st.session_state.get('last_designed_test_id'):
                        # Brena offers to measure the most recent test
                        test_id = st.session_state['last_designed_test_id']
                        test_name = st.session_state.get('last_designed_test_name', 'your test')
                        acknowledgment = f"""Thanks, {old_agent_name}! 👋 Hi there! I'm {new_agent_name}.

I see you just designed **"{test_name}"** ({test_id}). Would you like me to measure the results of that test?

Just say **"yes"** and I'll run the analysis, or tell me a different test ID if you'd like to measure something else."""
                    else:
                        acknowledgment = f"Thanks, {old_agent_name}! 👋 Hi there! I'm {new_agent_name}. How can I help you today?"
                    
                    with st.chat_message("assistant", avatar=new_avatar_path):
                        st.markdown(acknowledgment)
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': acknowledgment,
                        'agent': requested_agent,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Update current agent
                    st.session_state.current_agent = requested_agent
                    st.session_state.last_responding_agent = requested_agent
                    agent_transferred = True
                    
                else:
                    # Normal flow - get response from current agent
                    # First, get the response with a spinner (before deciding avatar)
                    with st.spinner(f"{AGENT_TITLES.get(current_agent, 'Agent')} is thinking..."):
                        response = run_agent(current_agent, prompt)
                    
                    # Detect if agent transfer occurred in response BEFORE rendering
                    new_agent = detect_agent_transfer(response, current_agent)
                    agent_transferred = new_agent != current_agent
                    
                    # Use the RESPONDING agent's avatar (new_agent if transferred)
                    responding_agent = new_agent if agent_transferred else current_agent
                    avatar_path = get_agent_avatar_path(responding_agent)
                    
                    # Now render with the correct avatar
                    with st.chat_message("assistant", avatar=avatar_path):
                        render_chat_response(response)
                    
                    if agent_transferred:
                        st.session_state.current_agent = new_agent
                        st.session_state.last_responding_agent = new_agent
                    
                    # Check if Brena completed a measurement or is showing results
                    # Update Advanced Statistics display accordingly
                    if responding_agent == 'brena':
                        detected_test = extract_test_id_from_response(response, prompt)
                        if detected_test:
                            st.session_state['adv_stats_displayed_test'] = detected_test
                    
                    # Add assistant message to history with RESPONDING agent
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'agent': responding_agent,  # Agent who actually responded
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Rerun to update UI with new agent if transferred
            if agent_transferred:
                st.rerun()
    
    # Return the workspace column for tab-specific content
    return col_workspace


# =============================================================================
# FILES VIEWER
# =============================================================================

def render_data_ribbon():
    """Render the always-on data ribbon with Master Test Tracker, Validation Files, and Datasets."""
    
    # Three clickable metric-card style links
    col1, col2, col3 = st.columns(3)
    
    # Determine which is selected for highlight styling
    master_selected = st.session_state.selected_file_view == 'master'
    validation_selected = st.session_state.selected_file_view == 'validation'
    datasets_selected = st.session_state.selected_file_view == 'datasets'
    
    with col1:
        # Metric card style for Master Test Tracker
        card_style = "opacity: 1;" if master_selected else "opacity: 0.85;"
        st.markdown(f"""
        <div class="metric-card" style="cursor: pointer; {card_style} padding: 0.8rem 1rem; text-align: center;">
            <div class="metric-card-header" style="font-size: 0.7rem;">📋 QUICK ACCESS</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0;">Master Test Tracker</div>
            <div class="metric-card-delta delta-neutral" style="font-size: 0.7rem;">All tests & results</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open", use_container_width=True, key="ribbon_master"):
            if master_selected:
                st.session_state.selected_file_view = None
            else:
                st.session_state.selected_file_view = 'master'
    
    with col2:
        card_style = "opacity: 1;" if validation_selected else "opacity: 0.85;"
        st.markdown(f"""
        <div class="metric-card" style="cursor: pointer; {card_style} padding: 0.8rem 1rem; text-align: center;">
            <div class="metric-card-header" style="font-size: 0.7rem;">✅ QUICK ACCESS</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0;">Validation Files</div>
            <div class="metric-card-delta delta-neutral" style="font-size: 0.7rem;">Download by test</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open", use_container_width=True, key="ribbon_validation"):
            if validation_selected:
                st.session_state.selected_file_view = None
            else:
                st.session_state.selected_file_view = 'validation'
    
    with col3:
        card_style = "opacity: 1;" if datasets_selected else "opacity: 0.85;"
        st.markdown(f"""
        <div class="metric-card" style="cursor: pointer; {card_style} padding: 0.8rem 1rem; text-align: center;">
            <div class="metric-card-header" style="font-size: 0.7rem;">📊 QUICK ACCESS</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0;">Datasets</div>
            <div class="metric-card-delta delta-neutral" style="font-size: 0.7rem;">Samples & reference</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open", use_container_width=True, key="ribbon_datasets"):
            if datasets_selected:
                st.session_state.selected_file_view = None
            else:
                st.session_state.selected_file_view = 'datasets'
    
    # Display content based on selection
    if st.session_state.selected_file_view:
        st.markdown("---")
        
        if st.session_state.selected_file_view == 'master':
            st.subheader("Master Test Tracker")
            st.caption("📋 Full tracking of every test — design, execution, and measurement results")
            try:
                df = load_csv('data/master_testing_doc.csv')
                # Show ALL columns - let users see everything we track
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
            except:
                st.info("No tests created yet.")
        
        elif st.session_state.selected_file_view == 'validation':
            # Two sections: Test Validation Files and Audience Files
            val_col1, val_col2 = st.columns(2)
            
            with val_col1:
                st.subheader("Test Validation Files")
                st.caption("Measurement outputs & model diagnostics")
                
                # Test selector dropdown
                try:
                    tests_df = load_csv('data/master_testing_doc.csv')
                    if not tests_df.empty and 'test_id' in tests_df.columns:
                        test_options = ["Select a test..."] + tests_df['test_id'].tolist()
                        selected_test = st.selectbox("Select Test:", test_options, key="validation_test_selector")
                        
                        if selected_test and selected_test != "Select a test...":
                            # Look for validation files for this test (GCS or local)
                            val_prefix = f"data/test_validation_files/{selected_test}"
                            folder_prefix = f"data/test_validation_files/{selected_test}_test_validation_files/"
                            
                            # List files from GCS or local
                            direct_files = list_files(f"data/test_validation_files/", ".csv")
                            direct_files = [f for f in direct_files if f.split('/')[-1].startswith(selected_test)]
                            folder_files = list_files(folder_prefix, ".csv")
                            
                            all_file_paths = direct_files + folder_files
                            
                            if all_file_paths:
                                st.markdown("**📥 Downloadable Files:**")
                                for file_path in all_file_paths:
                                    try:
                                        file_data = read_file_bytes(file_path)
                                        file_name = file_path.split('/')[-1]
                                        st.download_button(
                                            label=f"⬇️ {file_name}",
                                            data=file_data,
                                            file_name=file_name,
                                            mime="text/csv",
                                            key=f"dl_val_{file_name}"
                                        )
                                    except Exception as e:
                                        st.caption(f"Could not load {file_path.split('/')[-1]}")
                            else:
                                st.info(f"No validation files found for {selected_test}.")
                    else:
                        st.info("No tests available.")
                except:
                    st.info("Could not load test list.")
            
            with val_col2:
                st.subheader("Audience Files")
                st.caption("Test & control customer lists")
                
                # Test selector dropdown for audience files
                try:
                    tests_df = load_csv('data/master_testing_doc.csv')
                    if not tests_df.empty and 'test_id' in tests_df.columns:
                        test_options = ["Select a test..."] + tests_df['test_id'].tolist()
                        selected_aud_test = st.selectbox("Select Test:", test_options, key="audience_test_selector")
                        
                        if selected_aud_test and selected_aud_test != "Select a test...":
                            # Look for audience files in data/test_audience_files/ (GCS or local)
                            test_aud_path = f"data/test_audience_files/{selected_aud_test}_test_audience.csv"
                            control_aud_path = f"data/test_audience_files/{selected_aud_test}_control_audience.csv"
                            
                            has_files = False
                            
                            if file_exists(test_aud_path):
                                has_files = True
                                try:
                                    file_data = read_file_bytes(test_aud_path)
                                    file_name = f"{selected_aud_test}_test_audience.csv"
                                    st.download_button(
                                        label=f"⬇️ {file_name}",
                                        data=file_data,
                                        file_name=file_name,
                                        mime="text/csv",
                                        key=f"dl_aud_test_{selected_aud_test}"
                                    )
                                except:
                                    st.caption(f"Could not load test audience file")
                            
                            if file_exists(control_aud_path):
                                has_files = True
                                try:
                                    file_data = read_file_bytes(control_aud_path)
                                    file_name = f"{selected_aud_test}_control_audience.csv"
                                    st.download_button(
                                        label=f"⬇️ {file_name}",
                                        data=file_data,
                                        file_name=file_name,
                                        mime="text/csv",
                                        key=f"dl_aud_ctrl_{selected_aud_test}"
                                    )
                                except:
                                    st.caption(f"Could not load control audience file")
                            
                            if not has_files:
                                st.info(f"No audience files found for {selected_aud_test}.")
                    else:
                        st.info("No tests available.")
                except:
                    st.info("Could not load test list.")
        
        elif st.session_state.selected_file_view == 'datasets':
            st.subheader("Available Datasets")
            st.caption("Sample previews (500 rows) and full reference files")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**📈 CRM & Sales Data (Samples)**")
                
                # CRM Sales Data - 500 row sample
                try:
                    crm_path = 'data/crm_sales_cust_data/crm_sales_data_2025_H12026.csv'
                    if file_exists(crm_path):
                        df_sample = load_csv(crm_path, nrows=500)
                        csv_data = df_sample.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ crm_sales_data (500 rows)",
                            data=csv_data,
                            file_name="crm_sales_data_sample_500.csv",
                            mime="text/csv",
                            key="dl_crm_sales"
                        )
                except:
                    st.caption("crm_sales_data not available")
                
                # Customer File - 500 row sample
                try:
                    cust_path = 'data/crm_sales_cust_data/customer_file.csv'
                    if file_exists(cust_path):
                        df_sample = load_csv(cust_path, nrows=500)
                        csv_data = df_sample.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ customer_file (500 rows)",
                            data=csv_data,
                            file_name="customer_file_sample_500.csv",
                            mime="text/csv",
                            key="dl_customer"
                        )
                except:
                    st.caption("customer_file not available")
                
                # Order Line Items - 500 row sample
                try:
                    order_path = 'data/crm_sales_cust_data/order_line_items.csv'
                    if file_exists(order_path):
                        df_sample = load_csv(order_path, nrows=500)
                        csv_data = df_sample.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ order_line_items (500 rows)",
                            data=csv_data,
                            file_name="order_line_items_sample_500.csv",
                            mime="text/csv",
                            key="dl_orders"
                        )
                except:
                    st.caption("order_line_items not available")
            
            with col_b:
                st.markdown("**📦 Reference Data (Full Files)**")
                
                # Product Inventory - full
                try:
                    inv_path = 'data/product_inventory.csv'
                    if file_exists(inv_path):
                        file_data = read_file_bytes(inv_path)
                        st.download_button(
                            label="⬇️ product_inventory.csv",
                            data=file_data,
                            file_name="product_inventory.csv",
                            mime="text/csv",
                            key="dl_inventory"
                        )
                except:
                    st.caption("product_inventory not available")
                
                # Product SKU Info - full
                try:
                    sku_path = 'data/product_sku_info.csv'
                    if file_exists(sku_path):
                        file_data = read_file_bytes(sku_path)
                        st.download_button(
                            label="⬇️ product_sku_info.csv",
                            data=file_data,
                            file_name="product_sku_info.csv",
                            mime="text/csv",
                            key="dl_sku"
                        )
                except:
                    st.caption("product_sku_info not available")
                
                # Promo Calendar - full
                try:
                    promo_path = 'data/promo_calendar.csv'
                    if file_exists(promo_path):
                        file_data = read_file_bytes(promo_path)
                        st.download_button(
                            label="⬇️ promo_calendar.csv",
                            data=file_data,
                            file_name="promo_calendar.csv",
                            mime="text/csv",
                            key="dl_promo"
                        )
                except:
                    st.caption("promo_calendar not available")
        
        st.markdown("---")





# =============================================================================
# GUIDE WORKSPACE (Lira)
# =============================================================================

def render_guide_workspace():
    """Render the Guide workspace content (right side)."""
    
    # Company intro card
    with st.container():
        st.markdown("""
        <div class="ol-card">
            <h3>🏔️ Measurement Hub Guide</h3>
            <p>Outdoorsy Living is a premium outdoor apparel and gear company built on a culture of 
            <strong>test, learn, and optimize</strong>. We believe every marketing dollar should be accountable.</p>
            <hr>
            <p><strong>Meet Your Team:</strong></p>
            <ul>
                <li><strong>Lira</strong> — Measurement Hub Guide</li>
                <li><strong>Ketin</strong> — Test Design Specialist</li>
                <li><strong>Brena</strong> — Measurement Analyst</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Nudge Lira section
    st.markdown("#### 💬 Nudge Lira")
    st.caption("Hover for context, click to ask")
    
    # CSS for tooltip styling
    st.markdown("""
    <style>
    .nudge-container {
        position: relative;
        margin-bottom: 8px;
    }
    .nudge-btn {
        width: 100%;
        padding: 12px 16px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        text-align: left;
        cursor: pointer;
        font-size: 0.9rem;
        color: #495057;
        transition: all 0.2s ease;
        position: relative;
    }
    .nudge-btn:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-color: #2E7D32;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.15);
    }
    .nudge-btn::before {
        content: "💡";
        margin-right: 8px;
    }
    .nudge-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 100%;
        left: 0;
        right: 0;
        background: #1a1a2e;
        color: #fff;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 0.82rem;
        line-height: 1.5;
        z-index: 1000;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: opacity 0.2s ease, visibility 0.2s ease;
    }
    .nudge-tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 20px;
        border-width: 8px;
        border-style: solid;
        border-color: #1a1a2e transparent transparent transparent;
    }
    .nudge-container:hover .nudge-tooltip {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Define nudges with text and tooltips
    nudges = [
        {
            "text": "What's Outdoorsy Living, and why is it used here?",
            "tooltip": "Outdoorsy Living is a company made up out of thin air; a fictional outdoor-wear brand. The logo is kind of cool right? We created hyper-realistic order id level crm sales data, 100K reasonable customer profiles, 12 distinct products, product inventory, and promotional calendar. All used to demonstrate realistic test-and-measure workflows."
        },
        {
            "text": "What does the Hub automatically track during a test?",
            "tooltip": "I will be monitoring external factors like promotions, product inventory levels in this prototype. Easy can expand this to things like competitor movements, weather, ecommerce site health, consumer price index, and more."
        },
        {
            "text": "What measurement method do you use, how does it work? And why use this over other methodologies?",
            "tooltip": "We use Google's Causal Impact — a widely adopted method that estimates incremental lift by modeling what would've happened without the test."
        },
        {
            "text": "Do I still need to keep track of all the tests in-market, planned, or completed?",
            "tooltip": "Nope. All of it automatically gets logged and updated automatically in the master_testing_doc. E.g. it will capture things like the test name, hypothesis, test design setup inputs, and what the results of the test were."
        },
        {
            "text": "How can I be sure this is legit?",
            "tooltip": "Great question — I can show you the safeguards built in: pre- and post-period checks, validation files, version history, external-factor monitoring, and transparent logs so your data science team can verify everything themselves."
        }
    ]
    
    # Render nudge buttons with tooltips
    for i, nudge in enumerate(nudges):
        col1, col2 = st.columns([0.92, 0.08])
        with col1:
            st.markdown(f"""
            <div class="nudge-container">
                <div class="nudge-tooltip">{nudge['tooltip']}</div>
                <div class="nudge-btn">{nudge['text']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("→", key=f"nudge_btn_{i}", help="Ask Lira"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': nudge['text'],
                    'agent': None,
                    'timestamp': datetime.now().isoformat()
                })
                st.session_state.current_agent = 'lira'
                st.rerun()


# =============================================================================
# TEST DESIGN WORKSPACE (Ketin)
# =============================================================================

def render_test_design_workspace():
    """Render the Test Design workspace content (right side)."""
    
    config = st.session_state.test_design
    
    # Pre-test checklist at the top
    st.markdown("#### ✅ Pre-Test Checklist")
    st.caption("Ketin will verify each item as you design your test")
    
    checklist_items = [
        ("test_objective", "Test objective defined"),
        ("hypothesis", "Hypothesis documented"),
        ("kpi_selected", "Primary KPI selected"),
        ("split_method", "Split methodology chosen"),
        ("sample_size", "Sample size validated"),
        ("duration", "Test duration set"),
        ("pre_period_check", "Pre-period balance checked"),
        ("sign_off", "Ready for sign-off")
    ]
    
    for key, label in checklist_items:
        status = config.get('checklist', {}).get(key, False)
        icon = "✅" if status else "⬜"
        style = "checklist-done" if status else "checklist-pending"
        
        st.markdown(f"""
        <div class="{style} checklist-item">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ADVANCED STATISTICS WORKSPACE (Brena)
# =============================================================================

def render_advanced_stats_workspace():
    """Render the Advanced Statistics workspace content (right side).
    
    Displays measurement results from validation files:
    1. Metrics table (impact summary + model fit stats)
    2. Actual vs Predicted chart
    3. Pointwise effect chart
    4. Cumulative effect chart
    
    Auto-populates when Brena completes measurement.
    Users can also select a test manually from the dropdown.
    """
    
    st.markdown("#### 📊 Measurement Lab")
    st.caption("Deep-dive into measurement methodology and model diagnostics")
    
    # Test selector dropdown - allows manual selection
    try:
        tests_df = load_csv('data/master_testing_doc.csv')
        if not tests_df.empty and 'test_id' in tests_df.columns:
            test_options = tests_df['test_id'].tolist()
            
            # Get current selection from session state or default to first test
            current_selection = st.session_state.get('adv_stats_displayed_test', test_options[0] if test_options else None)
            
            # Find index of current selection
            try:
                default_idx = test_options.index(current_selection) if current_selection in test_options else 0
            except:
                default_idx = 0
            
            displayed_test = st.selectbox(
                "Select Test:", 
                test_options, 
                index=default_idx,
                key="ml_test_selector"
            )
            
            # Update session state when user changes selection
            st.session_state['adv_stats_displayed_test'] = displayed_test
        else:
            st.info("No tests available. Design and run a test first!")
            return
    except Exception as e:
        st.info("No tests found.")
        return
    
    if not displayed_test:
        return
    
    # Load validation files for the displayed test (GCS or local)
    validation_dir = f"data/test_validation_files/{displayed_test}_test_validation_files"
    
    if not file_exists(f"{validation_dir}/{displayed_test}_impact_summary.csv"):
        st.warning(f"Validation files not found for {displayed_test}. Has measurement been run?")
        return
    
    # Get test info from master doc
    try:
        tests_df = load_csv('data/master_testing_doc.csv')
        test_info = tests_df[tests_df['test_id'] == displayed_test].iloc[0] if not tests_df[tests_df['test_id'] == displayed_test].empty else None
    except:
        test_info = None
    
    # Display test header
    if test_info is not None:
        st.markdown(f"### {test_info.get('test_name', displayed_test)}")
        st.caption(f"Test ID: {displayed_test} | Status: {test_info.get('status', 'Unknown')}")
    else:
        st.markdown(f"### {displayed_test}")
    
    st.markdown("---")
    
    # ==========================================================================
    # 1. METRICS TABLE (Impact Summary + Model Fit Stats) - OUTDOORSY STYLE
    # ==========================================================================

    st.markdown("##### 📈 Key Metrics")

    # Local CSS just for this block (cards, typography, colors)
    st.markdown(
        """
        <style>
        .od-key-metrics-card {
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(135deg, #0b1720 0%, #020617 55%, #111827 100%);
            border: 1px solid rgba(148, 163, 184, 0.45);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.45);
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
        }

        .od-key-metrics-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }

        .od-key-metrics-title {
            font-size: 0.90rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #9ca3af;
        }

        .od-key-metrics-pill {
            font-size: 0.70rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.20rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.6);
            color: #e5e7eb;
            background: radial-gradient(circle at top left, rgba(52, 211, 153, 0.22), transparent 55%);
        }

        .od-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem 0.75rem;
        }

        .od-metric-full {
            grid-column: 1 / -1;
        }

        .od-metric-label {
            font-size: 0.70rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #9ca3af;
            margin-bottom: 0.15rem;
        }

        .od-metric-value {
            font-size: 1.05rem;
            font-weight: 600;
            line-height: 1.3;
        }

        .od-metric-value-small {
            font-size: 0.95rem;
            font-weight: 500;
        }

        .od-metric-positive {
            color: #4ade80;
        }

        .od-metric-negative {
            color: #f97373;
        }

        .od-metric-neutral {
            color: #e5e7eb;
        }

        .od-ci-text, .od-subtext {
            font-size: 0.75rem;
            color: #9ca3af;
        }

        .od-subtext {
            margin-top: 0.4rem;
        }

        .od-tag-row {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 0.35rem;
            margin-top: 0.45rem;
        }

        .od-tag {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            padding: 0.18rem 0.60rem;
            border-radius: 999px;
            border: 1px solid;
            display: inline-flex;
            align-items: center;
            gap: 0.30rem;
        }

        .od-tag-sig {
            border-color: rgba(52, 211, 153, 0.7);
            color: #6ee7b7;
            background: radial-gradient(circle at top left, rgba(16, 185, 129, 0.2), transparent 55%);
        }

        .od-tag-ns {
            border-color: rgba(249, 115, 22, 0.75);
            color: #fdba74;
            background: radial-gradient(circle at top left, rgba(249, 115, 22, 0.22), transparent 55%);
        }

        .od-tag-status-good {
            border-color: rgba(52, 211, 153, 0.7);
            color: #86efac;
            background: radial-gradient(circle at top left, rgba(22, 163, 74, 0.22), transparent 55%);
        }

        .od-tag-status-bad {
            border-color: rgba(248, 113, 113, 0.8);
            color: #fecaca;
            background: radial-gradient(circle at top left, rgba(220, 38, 38, 0.22), transparent 55%);
        }

        .od-fit-inline {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.5rem;
        }

        .od-fit-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9ca3af;
        }

        .od-fit-value {
            font-size: 0.94rem;
            font-weight: 500;
        }

        /* Equal height cards */
        .od-key-metrics-card {
            min-height: 280px;
            display: flex;
            flex-direction: column;
        }

        .od-key-metrics-card .od-metric-grid {
            flex-grow: 1;
        }

        .od-key-metrics-card .od-subtext {
            margin-top: auto;
            padding-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load impact summary (GCS or local)
    impact_file = f"{validation_dir}/{displayed_test}_impact_summary.csv"
    fit_file = f"{validation_dir}/{displayed_test}_model_fit_stats.csv"

    col_impact, col_fit = st.columns(2)

    # --------------------------------------------------------------------------
    # IMPACT SUMMARY CARD
    # --------------------------------------------------------------------------
    with col_impact:
        if file_exists(impact_file):
            try:
                impact_df = load_csv(impact_file)
                if not impact_df.empty:
                    row = impact_df.iloc[0]
                    avg_lift = row.get("avg_lift", 0)
                    rel_lift = row.get("relative_lift_pct", 0)
                    ci_lower = row.get("cred_int_lower", 0)
                    ci_upper = row.get("cred_int_upper", 0)
                    p_val = row.get("p_value_equivalent", "N/A")

                    # Determine significance using p-value (consistent with Business Insights tab)
                    is_sig = isinstance(p_val, (int, float)) and p_val < 0.05
                    sig_text = "Significant" if is_sig else "Not Significant"
                    sig_icon = "✅" if is_sig else "⚠️"

                    # Check if test is order-based or revenue-based from test description
                    test_desc = (
                        test_info.get("test_description", "").lower()
                        if test_info is not None
                        else ""
                    )
                    is_order_based = "order" in test_desc and "revenue" not in test_desc

                    # Formatting helpers
                    if pd.notna(avg_lift):
                        if is_order_based:
                            avg_lift_text = f"{avg_lift:,.0f}"
                        else:
                            avg_lift_text = f"${avg_lift:,.0f}"
                    else:
                        avg_lift_text = "N/A"

                    rel_lift_text = (
                        f"{rel_lift:.2f}%" if pd.notna(rel_lift) else "N/A"
                    )

                    if pd.notna(ci_lower):
                        if is_order_based:
                            ci_text = f"[{ci_lower:,.0f}, {ci_upper:,.0f}]"
                        else:
                            ci_text = f"[${ci_lower:,.0f}, ${ci_upper:,.0f}]"
                    else:
                        ci_text = "N/A"

                    if isinstance(p_val, (int, float)):
                        p_val_text = f"{p_val:.4f}"
                    else:
                        p_val_text = str(p_val)

                    # Color by sign of lift
                    if isinstance(avg_lift, (int, float)) and pd.notna(avg_lift):
                        if avg_lift > 0:
                            lift_class = "od-metric-positive"
                        elif avg_lift < 0:
                            lift_class = "od-metric-negative"
                        else:
                            lift_class = "od-metric-neutral"
                    else:
                        lift_class = "od-metric-neutral"
                    
                    # Get test name for the pill
                    test_name_pill = test_info.get("test_name", "Test") if test_info is not None else "Test"

                    sig_tag_class = 'od-tag-sig' if is_sig else 'od-tag-ns'
                    
                    impact_html = f'''<div class="od-key-metrics-card"><div class="od-key-metrics-header"><div class="od-key-metrics-title">Impact Summary</div><div class="od-key-metrics-pill">{test_name_pill}</div></div><div class="od-metric-grid"><div class="od-metric-full"><div class="od-metric-label">Average Daily Lift</div><div class="od-metric-value {lift_class}">{avg_lift_text}</div></div><div><div class="od-metric-label">Relative Lift</div><div class="od-metric-value-small od-metric-neutral">{rel_lift_text}</div></div><div><div class="od-metric-label">95% Credible Interval</div><div class="od-ci-text">{ci_text}</div></div><div><div class="od-metric-label">P-Value</div><div class="od-metric-value-small od-metric-neutral">{p_val_text}</div></div><div class="od-metric-full"><div class="od-tag-row"><span class="od-tag {sig_tag_class}"><span>{sig_icon}</span><span>{sig_text}</span></span></div></div></div><div class="od-subtext">Model-adjusted impact for this test. Values are per test period day and already account for underlying business trends and control performance.</div></div>'''

                    st.markdown(impact_html, unsafe_allow_html=True)
                else:
                    st.info("Impact summary not available")
            except Exception as e:
                st.warning(f"Could not load impact summary: {e}")
        else:
            st.info("Impact summary not available")

    # --------------------------------------------------------------------------
    # MODEL FIT STATISTICS CARD
    # --------------------------------------------------------------------------
    with col_fit:
        if file_exists(fit_file):
            try:
                fit_df = load_csv(fit_file)

                # Defaults
                mape_val = mae_val = rmse_val = r2_val = status_val = "N/A"

                for _, row in fit_df.iterrows():
                    metric_name = row.get("metric_name", "")
                    value = row.get("value", "N/A")

                    if metric_name == "MAPE":
                        mape_val = value
                    elif metric_name == "MAE":
                        mae_val = value
                    elif metric_name == "RMSE":
                        rmse_val = value
                    elif metric_name == "R_squared":
                        r2_val = value
                    elif metric_name == "convergence_status":
                        status_val = value

                # Threshold coloring
                def fmt_num(v, decimals=2):
                    return f"{v:.{decimals}f}" if isinstance(v, (int, float)) else v

                mape_display = (
                    f"{fmt_num(mape_val)}%" if mape_val != "N/A" else "N/A"
                )
                mae_display = (
                    f"{mae_val:,.0f}" if isinstance(mae_val, (int, float)) else mae_val
                )
                rmse_display = (
                    f"{rmse_val:,.0f}"
                    if isinstance(rmse_val, (int, float))
                    else rmse_val
                )
                r2_display = fmt_num(r2_val, 4)

                mape_good = isinstance(mape_val, (int, float)) and mape_val < 10
                r2_good = isinstance(r2_val, (int, float)) and r2_val > 0.8
                status_good = str(status_val).lower() == "converged"

                status_tag_class = "od-tag-status-good" if status_good else "od-tag-status-bad"
                status_icon = "🟢" if status_good else "🔴"
                mape_class = 'od-metric-positive' if mape_good else 'od-metric-neutral'
                r2_class = 'od-metric-positive' if r2_good else 'od-metric-neutral'

                fit_html = f'''<div class="od-key-metrics-card"><div class="od-key-metrics-header"><div class="od-key-metrics-title">Model Fit Statistics</div><div class="od-key-metrics-pill">Pre-test fit check</div></div><div class="od-metric-grid"><div class="od-metric-full od-fit-inline"><div class="od-fit-label">MAPE</div><div class="od-fit-value {mape_class}">{mape_display}</div></div><div class="od-fit-inline"><div class="od-fit-label">MAE</div><div class="od-fit-value od-metric-neutral">{mae_display}</div></div><div class="od-fit-inline"><div class="od-fit-label">RMSE</div><div class="od-fit-value od-metric-neutral">{rmse_display}</div></div><div class="od-fit-inline"><div class="od-fit-label">R²</div><div class="od-fit-value {r2_class}">{r2_display}</div></div><div class="od-metric-full"><div class="od-tag-row"><span class="od-tag {status_tag_class}"><span>{status_icon}</span><span>Status: {status_val}</span></span></div></div></div><div class="od-subtext">Lower MAPE and higher R² indicate a strong counterfactual fit. If fit degrades, treat impact estimates with caution or rerun the test with improved design.</div></div>'''

                st.markdown(fit_html, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Could not load model fit stats: {e}")
        else:
            st.info("Model fit stats not available")

    st.markdown("---")
    
    # ==========================================================================
    # 2. ACTUAL VS PREDICTED CHART
    # ==========================================================================
    st.markdown("##### 📊 Actual vs Predicted (Counterfactual)")
    
    timeseries_file = f"{validation_dir}/{displayed_test}_model_input_timeseries.csv"
    daily_effects_file = f"{validation_dir}/{displayed_test}_daily_effects.csv"
    
    if file_exists(timeseries_file) and file_exists(daily_effects_file):
        try:
            ts_df = load_csv(timeseries_file)
            effects_df = load_csv(daily_effects_file)
            
            ts_df['date'] = pd.to_datetime(ts_df['date'])
            effects_df['date'] = pd.to_datetime(effects_df['date'])
            
            # Get test period start date
            test_start = effects_df['date'].min()
            
            # Create actual vs predicted chart
            fig_actual = go.Figure()
            
            # Pre-period: treatment series
            pre_period = ts_df[ts_df['date'] < test_start]
            post_period_ts = ts_df[ts_df['date'] >= test_start]
            
            # Actual line (treatment series for pre + actual from effects for post)
            fig_actual.add_trace(go.Scatter(
                x=pre_period['date'], 
                y=pre_period['treatment_series'],
                name='Actual (Pre-Period)',
                line=dict(color='#1976D2', width=2)
            ))
            
            fig_actual.add_trace(go.Scatter(
                x=effects_df['date'], 
                y=effects_df['actual'],
                name='Actual (Test Period)',
                line=dict(color='#2E7D32', width=2)
            ))
            
            # Predicted/Counterfactual
            fig_actual.add_trace(go.Scatter(
                x=effects_df['date'], 
                y=effects_df['expected_counterfactual'],
                name='Predicted (Counterfactual)',
                line=dict(color='#FF9800', width=2, dash='dash')
            ))
            
            # Add test period shading
            fig_actual.add_vrect(
                x0=test_start, x1=effects_df['date'].max(),
                fillcolor="green", opacity=0.1,
                annotation_text="Test Period", annotation_position="top left"
            )
            
            fig_actual.update_layout(
                height=350, 
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="Revenue ($)"
            )
            st.plotly_chart(fig_actual, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not create actual vs predicted chart: {e}")
    else:
        st.info("Time series data not available for charting")
    
    st.markdown("---")
    
    # ==========================================================================
    # 3. POINTWISE EFFECT CHART
    # ==========================================================================
    st.markdown("##### 📉 Pointwise Effect (Daily Impact)")
    
    if file_exists(daily_effects_file):
        try:
            effects_df = load_csv(daily_effects_file)
            effects_df['date'] = pd.to_datetime(effects_df['date'])
            
            fig_point = go.Figure()
            
            # Effect line
            fig_point.add_trace(go.Scatter(
                x=effects_df['date'],
                y=effects_df['point_effect'],
                name='Point Effect',
                line=dict(color='#7B1FA2', width=2)
            ))
            
            # Confidence band
            fig_point.add_trace(go.Scatter(
                x=pd.concat([effects_df['date'], effects_df['date'][::-1]]),
                y=pd.concat([effects_df['upper_effect'], effects_df['lower_effect'][::-1]]),
                fill='toself',
                fillcolor='rgba(123, 31, 162, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))
            
            # Zero line
            fig_point.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_point.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="Effect ($)"
            )
            st.plotly_chart(fig_point, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not create pointwise chart: {e}")
    else:
        st.info("Daily effects data not available")
    
    st.markdown("---")
    
    # ==========================================================================
    # 4. CUMULATIVE EFFECT CHART
    # ==========================================================================
    st.markdown("##### 📈 Cumulative Effect (Total Impact Over Time)")
    
    if file_exists(daily_effects_file):
        try:
            effects_df = load_csv(daily_effects_file)
            effects_df['date'] = pd.to_datetime(effects_df['date'])
            
            # Calculate cumulative effects
            effects_df['cum_effect'] = effects_df['point_effect'].cumsum()
            effects_df['cum_lower'] = effects_df['lower_effect'].cumsum()
            effects_df['cum_upper'] = effects_df['upper_effect'].cumsum()
            
            fig_cum = go.Figure()
            
            # Cumulative effect line
            fig_cum.add_trace(go.Scatter(
                x=effects_df['date'],
                y=effects_df['cum_effect'],
                name='Cumulative Effect',
                line=dict(color='#00796B', width=2)
            ))
            
            # Confidence band
            fig_cum.add_trace(go.Scatter(
                x=pd.concat([effects_df['date'], effects_df['date'][::-1]]),
                y=pd.concat([effects_df['cum_upper'], effects_df['cum_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 121, 107, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))
            
            # Zero line
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_cum.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="Cumulative Effect ($)"
            )
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Show total cumulative effect
            total_effect = effects_df['cum_effect'].iloc[-1]
            st.metric("Total Cumulative Effect", f"${total_effect:,.0f}")
            
        except Exception as e:
            st.warning(f"Could not create cumulative chart: {e}")
    else:
        st.info("Daily effects data not available for cumulative chart")


def set_advanced_stats_test(test_id: str):
    """Helper function to set the displayed test in Advanced Statistics tab.
    Called by Brena when measurement completes or when user asks to see results.
    """
    st.session_state['adv_stats_displayed_test'] = test_id


# =============================================================================
# DATASET EXPLORER WORKSPACE (Standalone - Not part of agent workflow)
# =============================================================================

@st.cache_data(ttl=300)
def load_explorer_datasets():
    """Load all datasets for the explorer. Cached for 5 minutes."""
    datasets = {}
    
    # CRM Sales Data
    try:
        datasets['sales'] = gcs_loader.read_csv('data/crm_sales_cust_data/crm_sales_data_2025_H12026.csv')
        datasets['sales']['order_date'] = pd.to_datetime(datasets['sales']['order_date'])
    except Exception as e:
        print(f"Could not load sales data: {e}")
        datasets['sales'] = pd.DataFrame()
    
    # Customer File
    try:
        datasets['customers'] = gcs_loader.read_csv('data/crm_sales_cust_data/customer_file.csv')
    except Exception as e:
        print(f"Could not load customer data: {e}")
        datasets['customers'] = pd.DataFrame()
    
    # Order Line Items
    try:
        datasets['order_items'] = gcs_loader.read_csv('data/crm_sales_cust_data/order_line_items.csv')
    except Exception as e:
        print(f"Could not load order items: {e}")
        datasets['order_items'] = pd.DataFrame()
    
    # Product Inventory
    try:
        datasets['inventory'] = gcs_loader.read_csv('data/product_inventory.csv')
    except Exception as e:
        print(f"Could not load inventory: {e}")
        datasets['inventory'] = pd.DataFrame()
    
    # Product SKU Info
    try:
        datasets['products'] = gcs_loader.read_csv('data/product_sku_info.csv')
    except Exception as e:
        print(f"Could not load products: {e}")
        datasets['products'] = pd.DataFrame()
    
    return datasets


def get_dataset_schema_summary():
    """Get a summary of all datasets for the AI context."""
    datasets = load_explorer_datasets()
    
    schema_parts = []
    
    for name, df in datasets.items():
        if df.empty:
            continue
        
        schema_parts.append(f"""
**{name}** ({len(df):,} rows, {len(df.columns)} columns):
- Columns: {', '.join(df.columns.tolist())}
- Sample values: {df.head(2).to_dict('records')}
""")
    
    return "\n".join(schema_parts)


def process_explorer_query(user_query: str) -> dict:
    """
    Process a user query against the datasets using Gemini.
    Returns a dict with 'response', 'code', 'result_type', and 'data'.
    """
    datasets = load_explorer_datasets()
    schema_summary = get_dataset_schema_summary()
    
    # Build the prompt for Gemini
    system_prompt = f"""You are a data analysis assistant for Outdoorsy Living's marketing datasets.

AVAILABLE DATASETS (loaded as pandas DataFrames):
{schema_summary}

IMPORTANT DATASET RELATIONSHIPS:
- 'sales' contains order_id, customer_id, order_date, revenue, etc.
- 'customers' contains customer_id, customer_name, customer_attribute, dma, state, ltv_tier, etc.
- 'order_items' contains order_id, product_id, quantity, line_total
- 'products' contains product_id, product_name, description, price
- 'inventory' contains product_id, product_name, inventory_count, inventory_indicator

RULES:
1. Generate ONLY valid Python code using pandas, numpy, and plotly
2. The datasets dict is already available with keys: 'sales', 'customers', 'order_items', 'products', 'inventory'
3. Access dataframes like: datasets['sales'], datasets['customers'], etc.
4. For visualizations, use plotly.express (imported as px) or plotly.graph_objects (imported as go)
5. Store final results in a variable called `result`:
   - For tables: result = df (a pandas DataFrame)
   - For charts: result = fig (a plotly figure)
   - For text answers: result = "your answer string"
6. Keep code concise and efficient
7. Handle potential data issues gracefully (missing values, type conversions)
8. For date filtering, the sales data has dates from 2025-2026

RESPONSE FORMAT:
First, briefly explain what you're going to do (1-2 sentences).
Then provide the Python code in a ```python code block.

Example:
I'll calculate total revenue by customer attribute and show it as a bar chart.

```python
df = datasets['sales'].merge(datasets['customers'][['customer_id', 'customer_attribute']], on='customer_id')
revenue_by_attr = df.groupby('customer_attribute')['revenue'].sum().reset_index()
result = px.bar(revenue_by_attr, x='customer_attribute', y='revenue', title='Revenue by Customer Attribute')
```"""

    # Call Gemini
    try:
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=system_prompt)]),
                types.Content(role="user", parts=[types.Part(text=f"User query: {user_query}")])
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048
            )
        )
        
        response_text = response.text
        
        # Extract code from response
        code_match = re.search(r'```python\n(.*?)```', response_text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            explanation = response_text[:response_text.find('```python')].strip()
            
            # Execute the code safely
            try:
                # Set up execution environment
                exec_globals = {
                    'pd': pd,
                    'np': np,
                    'px': px,
                    'go': go,
                    'datasets': datasets,
                    'datetime': datetime,
                    'timedelta': timedelta
                }
                exec_locals = {}
                
                exec(code, exec_globals, exec_locals)
                
                result = exec_locals.get('result', None)
                
                if result is None:
                    return {
                        'response': explanation,
                        'code': code,
                        'result_type': 'text',
                        'data': "Code executed but no result was stored. Make sure to assign output to 'result' variable."
                    }
                
                # Determine result type
                if isinstance(result, pd.DataFrame):
                    return {
                        'response': explanation,
                        'code': code,
                        'result_type': 'dataframe',
                        'data': result
                    }
                elif hasattr(result, 'to_html'):  # Plotly figure
                    return {
                        'response': explanation,
                        'code': code,
                        'result_type': 'chart',
                        'data': result
                    }
                else:
                    return {
                        'response': explanation,
                        'code': code,
                        'result_type': 'text',
                        'data': str(result)
                    }
                    
            except Exception as exec_error:
                return {
                    'response': explanation,
                    'code': code,
                    'result_type': 'error',
                    'data': f"Error executing code: {str(exec_error)}"
                }
        else:
            # No code block, just return the text response
            return {
                'response': response_text,
                'code': None,
                'result_type': 'text',
                'data': response_text
            }
            
    except Exception as e:
        return {
            'response': None,
            'code': None,
            'result_type': 'error',
            'data': f"Error calling AI: {str(e)}"
        }


def render_dataset_explorer_workspace():
    """Render the Dataset Explorer workspace - completely isolated from agent workflow."""
    
    st.markdown("#### 🔍 Dataset Explorer")
    st.caption("Quick view of available datasets and a few static data-shape charts")

    # Quick info about available datasets (keep the selector-style summary)
    with st.expander("📦 Available Datasets", expanded=False):
        datasets = load_explorer_datasets()

        cols = st.columns(5)
        dataset_info = [
            ("sales", "Sales Data", "crm_sales_data_2025_H12026.csv"),
            ("customers", "Customers", "customer_file.csv"),
            ("order_items", "Order Items", "order_line_items.csv"),
            ("products", "Products", "product_sku_info.csv"),
            ("inventory", "Inventory", "product_inventory.csv")
        ]

        for col, (key, name, filename) in zip(cols, dataset_info):
            df = datasets.get(key, pd.DataFrame())
            with col:
                st.markdown(f"**{name}**")
                st.caption(f"{len(df):,} rows")

    # ---
    st.markdown("---")

    # STATIC CHARTS: quick-hitter summaries of data shape
    datasets = load_explorer_datasets()

    sales_df = datasets.get('sales', pd.DataFrame())
    order_items_df = datasets.get('order_items', pd.DataFrame())
    customers_df = datasets.get('customers', pd.DataFrame())

    # 1) Line Chart: Trending total orders by month (sales)
    st.markdown("**Orders Trend — Total Orders by Month**")
    if not sales_df.empty:
        try:
            tmp = sales_df.copy()
            # Normalize order date
            if 'order_date' in tmp.columns:
                tmp['order_date'] = pd.to_datetime(tmp['order_date'], errors='coerce')
            else:
                tmp['order_date'] = pd.to_datetime(tmp.iloc[:, 0], errors='coerce')

            # Determine order id column (common alternatives)
            order_id_cols = [c for c in ['order_id', 'unique_order_id', 'order_number', 'order_no', 'id'] if c in tmp.columns]
            if order_id_cols:
                order_col = order_id_cols[0]
            else:
                # fallback to first column that looks like an id
                order_col = tmp.columns[0]

            tmp = tmp.dropna(subset=['order_date'])
            if tmp.empty:
                st.info("Sales data has no parseable order dates to render trend.")
            else:
                tmp['month'] = tmp['order_date'].dt.to_period('M').dt.to_timestamp()
                orders_by_month = tmp.groupby('month')[order_col].nunique().reset_index(name='total_orders')
                if orders_by_month.empty:
                    st.info("No orders found to render trend.")
                else:
                    fig1 = px.line(orders_by_month, x='month', y='total_orders', markers=True, title='Total Orders by Month')
                    fig1.update_xaxes(title='Month')
                    fig1.update_yaxes(title='Total Orders')
                    st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render orders trend: {e}")
    else:
        st.info("Sales data not available to render orders trend.")

    # 2) Bar Chart: Top 5 products by revenue (order items)
    st.markdown("**Top 5 Products by Revenue**")
    if not order_items_df.empty:
        try:
            tmp_items = order_items_df.copy()
            # Identify revenue-like column
            revenue_cols = [c for c in ['line_total', 'price', 'revenue', 'amount', 'line_price'] if c in tmp_items.columns]
            if revenue_cols:
                rev_col = revenue_cols[0]
                tmp_items[rev_col] = pd.to_numeric(tmp_items[rev_col], errors='coerce').fillna(0)
                # Identify product id and optional product name
                prod_id_col = 'product_id' if 'product_id' in tmp_items.columns else tmp_items.columns[0]
                prod_rev = tmp_items.groupby(prod_id_col)[rev_col].sum().reset_index()

                # Resolve product names if available
                products_df = datasets.get('products', pd.DataFrame())
                if not products_df.empty and 'product_id' in products_df.columns and 'product_name' in products_df.columns:
                    prod_rev = prod_rev.merge(products_df[['product_id', 'product_name']], left_on=prod_id_col, right_on='product_id', how='left')
                    prod_rev['label'] = prod_rev['product_name'].fillna(prod_rev[prod_id_col].astype(str))
                else:
                    prod_rev['label'] = prod_rev[prod_id_col].astype(str)

                # Determine revenue column name in aggregated frame
                agg_rev_col = rev_col
                top5 = prod_rev.sort_values(agg_rev_col, ascending=False).head(5)
                fig2 = px.bar(top5, x='label', y=agg_rev_col, title='Top 5 Products by Revenue', labels={'label':'Product', agg_rev_col:'Revenue'})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info('Order items do not contain a revenue/price column to calculate product revenue.')
        except Exception as e:
            st.warning(f"Could not render top products chart: {e}")
    else:
        st.info("Order items data not available to render product revenue chart.")

    # 3) Bar Chart: Top 5 customer attributes by distinct customer count
    st.markdown("**Top 5 Customer Attributes (Distinct Customers)**")
    if not customers_df.empty and 'customer_attribute' in customers_df.columns:
        try:
            cust_counts = customers_df.groupby('customer_attribute')['customer_id'].nunique().reset_index(name='distinct_customers')
            top_attrs = cust_counts.sort_values('distinct_customers', ascending=False).head(5)
            fig3 = px.bar(top_attrs, x='customer_attribute', y='distinct_customers', title='Top 5 Customer Attributes by Distinct Customers', labels={'customer_attribute':'Attribute','distinct_customers':'Distinct Customers'})
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render customer attribute chart: {e}")
    else:
        st.info("Customer data not available to render attribute distribution.")
    
    # Display results
    st.markdown("---")
    
    if st.session_state.explorer_results:
        result = st.session_state.explorer_results
        
        # Show AI explanation
        if result.get('response'):
            st.markdown(f"**🤖 Analysis:** {result['response']}")
        
        # Show result based on type
        result_type = result.get('result_type')
        data = result.get('data')
        
        if result_type == 'dataframe' and isinstance(data, pd.DataFrame):
            st.dataframe(data, use_container_width=True, height=400)
            
            # Download button
            csv = data.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "explorer_results.csv",
                "text/csv",
                key="download_explorer_csv"
            )
            
        elif result_type == 'chart':
            st.plotly_chart(data, use_container_width=True)
            
        elif result_type == 'error':
            st.error(f"⚠️ {data}")
            if result.get('code'):
                with st.expander("View attempted code"):
                    st.code(result['code'], language='python')
                    
        elif result_type == 'text':
            st.info(data)
        
        # Show code (collapsed)
        if result.get('code'):
            with st.expander("📝 View generated code"):
                st.code(result['code'], language='python')
    
    # Show chat history (last 5)
    if st.session_state.explorer_chat_history:
        st.markdown("---")
        with st.expander("📜 Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.explorer_chat_history[-5:])):
                st.markdown(f"**{i+1}.** {item['query']}")
                st.caption(item['timestamp'][:19])


# =============================================================================
# BUSINESS IMPACT WORKSPACE (Brena)
# =============================================================================

def render_business_impact_workspace():
    """Render the Business Impact workspace content (right side)."""
    
    st.markdown("#### 💼 Summary Impact")
    st.caption("The executive summary — what you need to make a decision")
    
    # Test selector
    try:
        tests_df = load_csv('data/master_testing_doc.csv')
        if not tests_df.empty and 'test_id' in tests_df.columns:
            test_options = tests_df['test_id'].tolist()
            selected_test = st.selectbox("Select Test:", test_options, key="bi_test_selector")
        else:
            st.info("No tests available. Design and run a test first!")
            return
    except:
        st.info("No tests found.")
        return
    
    # Load measurement results for selected test (GCS or local)
    impact_file = f"data/test_validation_files/{selected_test}_test_validation_files/{selected_test}_impact_summary.csv"
    diagnostics_file = f"data/test_validation_files/{selected_test}_model_diagnostics.json"
    
    # Get test info to determine if order-based or revenue-based
    test_row = tests_df[tests_df['test_id'] == selected_test]
    test_desc = test_row.iloc[0].get('test_description', '').lower() if not test_row.empty else ''
    is_order_based = 'order' in test_desc and 'revenue' not in test_desc
    
    # Check if measurement has been run
    has_results = file_exists(impact_file) or file_exists(diagnostics_file)
    
    if not has_results:
        st.info("🔬 Run measurement in the **Measurement Lab** tab to see results for this test.")
        return
    
    # Load the results
    incr_revenue = 0.0
    lift_pct = 0.0
    p_value = 1.0
    actual_revenue = 0.0
    predicted_revenue = 0.0
    ci_lower = 0.0
    ci_upper = 0.0
    
    try:
        # Try impact_summary.csv first
        if file_exists(impact_file):
            impact_df = load_csv(impact_file)
            if not impact_df.empty:
                row = impact_df.iloc[0]
                incr_revenue = row.get('avg_lift', row.get('cum_lift', 0)) or 0
                lift_pct = row.get('relative_lift_pct', 0) or 0
                p_value = row.get('p_value_equivalent', 1.0) or 1.0
                ci_lower = row.get('cred_int_lower', 0) or 0
                ci_upper = row.get('cred_int_upper', 0) or 0
        
        # Also try model_diagnostics.json for additional data
        if file_exists(diagnostics_file):
            diag_content = gcs_loader.read_text(diagnostics_file)
            diag = json.loads(diag_content)
            if 'p_value' in diag:
                p_value = diag['p_value']
            # Parse summary_stats if available
            if 'summary_stats' in diag and isinstance(diag['summary_stats'], str):
                # Extract values from the summary string
                lines = diag['summary_stats'].strip().split('\n')
                if len(lines) >= 2:
                    # Parse the cumulative line (last one)
                    parts = lines[-1].split()
                    if len(parts) >= 10:
                        actual_revenue = float(parts[1])
                        predicted_revenue = float(parts[2])
                        incr_revenue = float(parts[6])
                        lift_pct = float(parts[10]) * 100  # rel_effect is decimal
    except Exception as e:
        pass  # Use defaults if loading fails
    
    # Calculate confidence from p-value
    confidence = max(0, min(100, (1 - p_value) * 100))
    is_significant = p_value < 0.05
    is_positive = lift_pct > 0
    
    st.markdown("---")
    
    # Key metrics in 2x2 grid for narrower workspace
    st.markdown("##### 📊 Key Results")
    
    # Row 1: Impact metrics (Incremental Revenue + Impact Range)
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        delta_class = "delta-positive" if incr_revenue > 0 else "delta-negative" if incr_revenue < 0 else "delta-neutral"
        lift_sign = '+' if lift_pct > 0 else ''
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">Incremental Revenue</div>
            <div class="metric-card-value">${incr_revenue:,.0f}</div>
            <div class="metric-card-delta {delta_class}">{lift_sign}{lift_pct:.1f}% lift</div>
        </div>
        """, unsafe_allow_html=True)
    
    with row1_col2:
        # Impact Range card - shows CI range on one line, neutral color
        # Format based on metric type (orders vs revenue)
        if is_order_based:
            ci_display = f"{ci_lower:,.0f} — {ci_upper:,.0f}"
        else:
            ci_display = f"${ci_lower:,.0f} — ${ci_upper:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">Impact Range</div>
            <div class="metric-card-value">{ci_display}</div>
            <div class="metric-card-delta delta-neutral">95% confidence interval</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")  # Spacing
    
    # Row 2: Statistical metrics (p-value + Confidence)
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        sig_text = "Significant" if is_significant else "Not significant"
        sig_class = "delta-positive" if is_significant else "delta-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">p-value</div>
            <div class="metric-card-value">{p_value:.4f}</div>
            <div class="metric-card-delta {sig_class}">{sig_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with row2_col2:
        conf_class = "delta-positive" if confidence >= 95 else "delta-neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-header">Confidence</div>
            <div class="metric-card-value">{confidence:.1f}%</div>
            <div class="metric-card-delta {conf_class}">Statistical confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendation section with conditional styling
    st.markdown("##### 💡 Recommendation")
    
    if lift_pct > 0 and is_significant:
        # Strong positive result - GO
        st.markdown("""
        <div class="results-highlight">
            <h3>✅ Scale This Campaign</h3>
            <p>Strong positive results with high statistical confidence. The test drove significant incremental lift. Consider expanding to additional markets or increasing investment.</p>
        </div>
        """, unsafe_allow_html=True)
    elif lift_pct > 0 and not is_significant:
        # Positive direction but not significant - CAUTIOUS
        st.markdown("""
        <div class="results-highlight results-highlight-caution">
            <h3>📊 Cautiously Optimistic</h3>
            <p>Positive direction but results are not statistically significant. Consider extending the test duration or increasing sample size for higher confidence before scaling.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Negative or no lift - REVIEW
        st.markdown("""
        <div class="results-highlight results-highlight-negative">
            <h3>⚠️ Review Strategy</h3>
            <p>No significant positive lift detected. Consider reviewing campaign creative, targeting, channel mix, or test design before making investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize OTel tracing
    initialize_otel_tracer()
    
    # Render sidebar first (always on)
    render_sidebar()
    
    # Render OPEN ME dialog if triggered
    render_open_me_dialog()
    
    # Always-on data ribbon (Master Testing Doc, Validation Files, Datasets)
    render_data_ribbon()
    
    # Render persistent chat (left) and get workspace container (right)
    workspace_col = render_persistent_chat()
    
    # Workspace content (right side) with tabs
    with workspace_col:
        st.markdown("### 🛠️ Workspace")
        
        # Tab names and their indices
        tab_names = [
            "🏔️ Guide",
            "🎯 Test Design Lab", 
            "📊 Measurement Lab",
            "💼 Summary Impact",
            "🔍 Dataset Explorer"
        ]
        tab_keys = ["guide", "design", "measurement", "impact", "explorer"]
        
        # Get current tab from session state (persists across reruns)
        if 'workspace_tab' not in st.session_state:
            st.session_state.workspace_tab = "guide"
        
        # Create tab selector that persists
        selected_tab = st.radio(
            "Workspace View",
            tab_keys,
            format_func=lambda x: tab_names[tab_keys.index(x)],
            horizontal=True,
            key="workspace_tab_selector",
            index=tab_keys.index(st.session_state.workspace_tab),
            label_visibility="collapsed"
        )
        
        # Update session state
        st.session_state.workspace_tab = selected_tab
        
        st.markdown("---")
        
        # Render the selected workspace
        if selected_tab == "guide":
            render_guide_workspace()
        elif selected_tab == "design":
            render_test_design_workspace()
        elif selected_tab == "measurement":
            render_advanced_stats_workspace()
        elif selected_tab == "impact":
            render_business_impact_workspace()
        elif selected_tab == "explorer":
            render_dataset_explorer_workspace()


if __name__ == "__main__":
    main()
