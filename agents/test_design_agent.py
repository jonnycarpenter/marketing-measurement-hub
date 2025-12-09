"""
Ketin Vale - Test Design Specialist Agent

Ketin helps marketing leaders design bulletproof test and control experiments.
He's methodical, detail-oriented, and passionate about proper experimental design.
"""
import logging
from pathlib import Path
from google.adk.agents import LlmAgent

from utils.model_manager import get_dynamic_model
from utils.gcs_loader import gcs_loader

from .tools import (
    get_customer_data_summary,
    get_existing_tests,
    get_test_details,
    calculate_sample_size,
    create_random_split,
    create_geographic_split,
    save_current_test_design,
    get_current_split_status,
    search_causal_impact_knowledge,
    check_promo_calendar,
    # Visualization tools
    generate_test_design_chart,
    get_available_chart_options,
    generate_pre_period_balance_charts,
    # Shared context tools
    read_shared_context,
    update_shared_context,
    list_session_artifacts,
    # Checklist tool
    update_test_checklist
)

logger = logging.getLogger(__name__)

# Dynamic model selection - starts with gemini-3-pro-preview, falls back to gemini-2.5-pro on token limits
# Model manager handles automatic fallback when context limits are hit
AGENT_NAME = "ketin"

def get_model():
    """Get current model with dynamic fallback support."""
    return get_dynamic_model(AGENT_NAME)

# Load system instruction from prompt file (GCS in prod, local in dev)
TEST_DESIGN_INSTRUCTION = gcs_loader.read_text("agents/agent_prompts/ketin_test_design_prompt.txt")

# Agent tools list (shared between instances)
TEST_DESIGN_TOOLS = [
    get_customer_data_summary,
    get_existing_tests,
    get_test_details,
    calculate_sample_size,
    create_random_split,
    create_geographic_split,
    save_current_test_design,
    get_current_split_status,
    search_causal_impact_knowledge,
    check_promo_calendar,
    # Visualization tools
    generate_test_design_chart,
    get_available_chart_options,
    generate_pre_period_balance_charts,
    # Shared context tools
    read_shared_context,
    update_shared_context,
    list_session_artifacts,
    # Checklist tool
    update_test_checklist
]


def create_test_design_agent(model: str = None) -> LlmAgent:
    """Create a test design agent with the specified or current dynamic model."""
    current_model = model or get_model()
    logger.info(f"Creating Ketin agent with model: {current_model}")
    return LlmAgent(
        name="Ketin_Test_Design",
        model=current_model,
        description="Ketin Vale, the Test Design Specialist at Outdoorsy Living. Methodical and passionate about proper experimental design. Guides users through setting up bulletproof test/control experiments.",
        instruction=TEST_DESIGN_INSTRUCTION,
        tools=TEST_DESIGN_TOOLS,
        output_key="test_design_result"
    )


# Create the default Test Design Agent (for backwards compatibility)
test_design_agent = create_test_design_agent()


def get_test_design_agent(force_refresh: bool = False) -> LlmAgent:
    """Get the test design agent instance, optionally refreshing with current model."""
    global test_design_agent
    if force_refresh:
        test_design_agent = create_test_design_agent()
    return test_design_agent
