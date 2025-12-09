"""
Brena Colette - Measurement Analyst Agent

Brena helps marketing leaders understand test results using CausalImpact.
She's brilliant at translating complex statistics into clear business decisions.
"""
import logging
from pathlib import Path
from google.adk.agents import LlmAgent

from utils.model_manager import get_dynamic_model
from utils.gcs_loader import gcs_loader

from .tools import (
    get_existing_tests,
    get_test_details,
    run_causal_impact_analysis,
    get_measurement_summary,
    update_test_status,
    search_causal_impact_knowledge,
    # Validation data analysis tools
    get_test_validation_files,
    get_validation_file_data,
    get_pre_period_balance_check,
    get_model_diagnostics,
    get_robustness_analysis,
    get_daily_effects_analysis,
    # Visualization tools
    get_causal_impact_charts,
    # Shared context tools
    read_shared_context,
    update_shared_context,
    list_session_artifacts
)

logger = logging.getLogger(__name__)

# Dynamic model selection - starts with gemini-3-pro-preview, falls back to gemini-2.5-pro on token limits
# Model manager handles automatic fallback when context limits are hit
AGENT_NAME = "brena"

def get_model():
    """Get current model with dynamic fallback support."""
    return get_dynamic_model(AGENT_NAME)

# Load system instruction from prompt file (GCS in prod, local in dev)
MEASUREMENT_INSTRUCTION = gcs_loader.read_text("agents/agent_prompts/brena_measurement_prompt.txt")

# Agent tools list (shared between instances)
MEASUREMENT_TOOLS = [
    get_existing_tests,
    get_test_details,
    run_causal_impact_analysis,
    get_measurement_summary,
    update_test_status,
    search_causal_impact_knowledge,
    # Validation data analysis tools
    get_test_validation_files,
    get_validation_file_data,
    get_pre_period_balance_check,
    get_model_diagnostics,
    get_robustness_analysis,
    get_daily_effects_analysis,
    # Visualization tools
    get_causal_impact_charts,
    # Shared context tools
    read_shared_context,
    update_shared_context,
    list_session_artifacts
]


def create_measurement_agent(model: str = None) -> LlmAgent:
    """Create a measurement agent with the specified or current dynamic model."""
    current_model = model or get_model()
    logger.info(f"Creating Brena agent with model: {current_model}")
    return LlmAgent(
        name="Brena_Measurement_Analyst",
        model=current_model,
        description="Brena Colette, the Measurement Analyst at Outdoorsy Living. Brilliant at translating complex statistics into clear business decisions. Runs CausalImpact analysis and provides GO/NO-GO recommendations.",
        instruction=MEASUREMENT_INSTRUCTION,
        tools=MEASUREMENT_TOOLS,
        output_key="measurement_result"
    )


# Create the default Measurement Agent (for backwards compatibility)
measurement_agent = create_measurement_agent()


def get_measurement_agent(force_refresh: bool = False) -> LlmAgent:
    """Get the measurement agent instance, optionally refreshing with current model."""
    global measurement_agent
    if force_refresh:
        measurement_agent = create_measurement_agent()
    return measurement_agent
