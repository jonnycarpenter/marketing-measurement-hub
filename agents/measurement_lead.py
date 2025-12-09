"""
Lira Maven - Measurement Lead (Orchestrator) Agent

Lira is the main coordinator for the Outdoorsy Living marketing measurement hub.
She acts as the primary interface for marketing leaders, routing requests
to specialized agents (Ketin for Test Design, Brena for Measurement).
"""
import logging
from pathlib import Path
from google.adk.agents import LlmAgent

from utils.model_manager import get_dynamic_model
from utils.gcs_loader import gcs_loader

from .test_design_agent import create_test_design_agent
from .measurement_agent import create_measurement_agent
from .tools import (
    get_customer_data_summary,
    get_crm_sales_data_summary,
    get_order_line_items_summary,
    get_existing_tests,
    get_test_details,
    search_measurement_methodology,
    compare_measurement_methodologies,
    # Shared context tools
    read_shared_context,
    update_shared_context
)

logger = logging.getLogger(__name__)

# Dynamic model selection - starts with gemini-3-pro-preview, falls back to gemini-2.5-pro on token limits
# Model manager handles automatic fallback when context limits are hit
# Lira uses flash for faster routing responses
AGENT_NAME = "lira"

def get_model():
    """Get current model with dynamic fallback support."""
    return get_dynamic_model(AGENT_NAME)

# Load system instruction from prompt file (GCS in prod, local in dev)
ORCHESTRATOR_INSTRUCTION = gcs_loader.read_text("agents/agent_prompts/lira_guided_tour_prompt.txt")

# Agent tools list (shared between instances)
ORCHESTRATOR_TOOLS = [
    get_customer_data_summary,
    get_crm_sales_data_summary,
    get_order_line_items_summary,
    get_existing_tests,
    get_test_details,
    search_measurement_methodology,
    compare_measurement_methodologies,
    # Shared context tools
    read_shared_context,
    update_shared_context
]


def create_measurement_lead_agent(model: str = None) -> LlmAgent:
    """Create a measurement lead agent with the specified or current dynamic model.
    
    Note: Sub-agents must be created fresh each time since ADK agents can only have one parent.
    Reusing agent instances across parent agents causes 'already has a parent' errors.
    """
    current_model = model or get_model()
    logger.info(f"Creating Lira agent with model: {current_model}")
    
    # Create fresh sub-agent instances - each agent can only have one parent
    ketin = create_test_design_agent(current_model)
    brena = create_measurement_agent(current_model)
    
    return LlmAgent(
        name="Lira_Guided_Tour",
        model=current_model,
        description="Lira, the Measurement Hub guide at Outdoorsy Living. Welcomes users to the Measurement Hub, provides data overviews, and routes to specialists (Ketin for test design, Brena for measurement).",
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=ORCHESTRATOR_TOOLS,
        sub_agents=[ketin, brena]
    )


# Create the default Orchestrator Agent (for backwards compatibility)
measurement_lead_agent = create_measurement_lead_agent()


def get_measurement_lead_agent(force_refresh: bool = False) -> LlmAgent:
    """Get the measurement lead orchestrator agent, optionally refreshing with current model."""
    global measurement_lead_agent
    if force_refresh:
        measurement_lead_agent = create_measurement_lead_agent()
    return measurement_lead_agent


# Also export as root_agent for ADK compatibility
root_agent = measurement_lead_agent
