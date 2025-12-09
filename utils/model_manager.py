"""
Dynamic Model Manager for ADK Agents

Provides intelligent model selection with automatic fallback when token limits are hit.

Agent Model Assignments:
    - Ketin (Test Design): Gemini 2.5 Pro → Gemini 2.5 Flash fallback
    - Lira (Guide): Gemini 2.5 Pro → Gemini 2.5 Flash fallback  
    - Brena (Measurement): Gemini 2.5 Pro → Gemini 2.5 Flash fallback
    
Note: Claude models (Opus 4.5, Sonnet 4.5) are available in AVAILABLE_MODELS for future use.
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Model provider identifiers."""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    display_name: str
    provider: ModelProvider
    max_context_tokens: int
    temperature: float = 0.0
    is_fallback: bool = False


@dataclass
class AgentModelConfig:
    """Model configuration for a specific agent."""
    primary_model: str
    fallback_model: str
    description: str


@dataclass
class ModelState:
    """Tracks the state of model usage per session."""
    current_model: str
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    fallback_time: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
# NOTE: Model strings from Google Vertex AI Model Garden (Dec 2025)

CLAUDE_OPUS_4_5 = "claude-opus-4-5@20251101"
CLAUDE_SONNET_4_5 = "claude-sonnet-4-5@20250929"
GEMINI_3_PRO = "gemini-3-pro-preview"
GEMINI_2_5_PRO = "gemini-2.5-pro"
GEMINI_2_5_FLASH = "gemini-2.5-flash"

AVAILABLE_MODELS = {
    CLAUDE_OPUS_4_5: ModelConfig(
        name=CLAUDE_OPUS_4_5,
        display_name="Claude Opus 4.5",
        provider=ModelProvider.ANTHROPIC,
        max_context_tokens=40000,
    ),
    CLAUDE_SONNET_4_5: ModelConfig(
        name=CLAUDE_SONNET_4_5,
        display_name="Claude Sonnet 4.5",
        provider=ModelProvider.ANTHROPIC,
        max_context_tokens=40000,
    ),
    GEMINI_3_PRO: ModelConfig(
        name=GEMINI_3_PRO,
        display_name="Gemini 3 Pro",
        provider=ModelProvider.GOOGLE,
        temperature=0.5,
        max_context_tokens=1000000,
        is_fallback=True,
    ),
    GEMINI_2_5_PRO: ModelConfig(
        name=GEMINI_2_5_PRO,
        display_name="Gemini 2.5 Pro",
        provider=ModelProvider.GOOGLE,
        temperature=0.4,
        max_context_tokens=1000000,
        is_fallback=True,
    ),
    GEMINI_2_5_FLASH: ModelConfig(
        name=GEMINI_2_5_FLASH,
        display_name="gemini-2.5-flash",
        provider=ModelProvider.GOOGLE,
        temperature=0.4,
        max_context_tokens=1000000,
        is_fallback=True,
    ),
}

# =============================================================================
# AGENT MODEL ASSIGNMENTS
# =============================================================================
# Define which model each agent uses primarily and falls back to

AGENT_MODEL_ASSIGNMENTS: Dict[str, AgentModelConfig] = {
    "ketin": AgentModelConfig(
        primary_model=GEMINI_2_5_PRO,
        fallback_model=GEMINI_2_5_FLASH,
        description="Test Design Agent - Uses Gemini 3 Pro for rigorous statistical reasoning"
    ),
    "lira": AgentModelConfig(
        primary_model=GEMINI_2_5_PRO,
        fallback_model=GEMINI_2_5_FLASH,
        description="Guide Agent - Uses Gemini 2.5 Pro for clear explanations"
    ),
    "brena": AgentModelConfig(
        primary_model=GEMINI_2_5_PRO,
        fallback_model=GEMINI_2_5_FLASH,
        description="Measurement Agent - Uses Gemini 2.5 Pro for analysis and recommendations"
    ),
}

# Default config for any agent not explicitly defined
DEFAULT_AGENT_CONFIG = AgentModelConfig(
    primary_model=GEMINI_2_5_PRO,
    fallback_model=GEMINI_2_5_FLASH,
    description="Default agent configuration"
)

# =============================================================================
# ERROR PATTERNS
# =============================================================================
# Error patterns that indicate we should fall back to a different model

TOKEN_LIMIT_ERROR_PATTERNS = [
    "token limit",
    "context length",
    "maximum context",
    "too many tokens",
    "context window",
    "input too long",
    "exceeds the limit",
    "request too large",
    "payload too large",
    "content too long",
    "resource_exhausted",
    "RESOURCE_EXHAUSTED",
]

RATE_LIMIT_ERROR_PATTERNS = [
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
    "quota_exceeded",
]

# Combine all patterns that should trigger fallback
FALLBACK_ERROR_PATTERNS = TOKEN_LIMIT_ERROR_PATTERNS + RATE_LIMIT_ERROR_PATTERNS


class DynamicModelManager:
    """
    Manages dynamic model selection with automatic fallback.
    
    Supports per-agent model assignments:
        - Ketin: Gemini 2.5 Pro (primary) → Gemini 2.5 Flash (fallback)
        - Lira: Gemini 2.5 Pro (primary) → Gemini 2.5 Flash (fallback)
        - Brena: Gemini 2.5 Pro (primary) → Gemini 2.5 Flash (fallback)
    
    Usage:
        manager = DynamicModelManager()
        model = manager.get_current_model("ketin")
        
        # If an error occurs:
        if manager.should_fallback(error):
            manager.trigger_fallback("ketin", error)
            model = manager.get_current_model("ketin")
    """
    
    def __init__(self):
        self._session_states: Dict[str, ModelState] = {}
        self._models = AVAILABLE_MODELS
        self._agent_configs = AGENT_MODEL_ASSIGNMENTS
        
    def _get_agent_config(self, agent_name: str) -> AgentModelConfig:
        """Get the model configuration for an agent."""
        agent_key = agent_name.lower()
        return self._agent_configs.get(agent_key, DEFAULT_AGENT_CONFIG)
    
    def get_current_model(self, agent_name: str) -> str:
        """Get the current model to use for an agent."""
        state = self._get_or_create_state(agent_name)
        return state.current_model
    
    def get_model_display_info(self, agent_name: str) -> Dict[str, Any]:
        """Get display info about current model state for an agent."""
        state = self._get_or_create_state(agent_name)
        agent_config = self._get_agent_config(agent_name)
        model_config = self._models.get(state.current_model)
        
        return {
            "current_model": state.current_model,
            "display_name": model_config.display_name if model_config else state.current_model,
            "provider": model_config.provider.value if model_config else "unknown",
            "is_primary": state.current_model == agent_config.primary_model,
            "fallback_triggered": state.fallback_triggered,
            "fallback_reason": state.fallback_reason,
            "primary_model": agent_config.primary_model,
            "fallback_model": agent_config.fallback_model,
        }
    
    def should_fallback(self, error: Exception) -> bool:
        """Check if an error indicates we should fall back to a different model."""
        error_str = str(error).lower()
        return any(pattern.lower() in error_str for pattern in FALLBACK_ERROR_PATTERNS)
    
    def trigger_fallback(self, agent_name: str, error: Exception) -> bool:
        """
        Trigger fallback to the fallback model for this agent.
        
        Returns True if fallback was triggered, False if no fallback available.
        """
        state = self._get_or_create_state(agent_name)
        agent_config = self._get_agent_config(agent_name)
        
        # Already on fallback model
        if state.current_model == agent_config.fallback_model:
            logger.warning(
                f"[{agent_name}] Already on fallback model ({agent_config.fallback_model}), "
                "cannot fall back further"
            )
            state.error_count += 1
            state.last_error = str(error)
            return False
        
        # Switch to fallback model
        old_model = state.current_model
        state.current_model = agent_config.fallback_model
        state.fallback_triggered = True
        state.fallback_reason = str(error)[:200]
        state.fallback_time = datetime.now()
        
        logger.info(
            f"[{agent_name}] Triggered model fallback: {old_model} -> {agent_config.fallback_model} "
            f"(reason: {state.fallback_reason[:50]}...)"
        )
        
        return True
    
    def reset_to_primary(self, agent_name: str) -> None:
        """Reset an agent back to its primary model."""
        state = self._get_or_create_state(agent_name)
        agent_config = self._get_agent_config(agent_name)
        
        if state.current_model != agent_config.primary_model:
            logger.info(
                f"[{agent_name}] Resetting to primary model: {agent_config.primary_model}"
            )
            state.current_model = agent_config.primary_model
            state.fallback_triggered = False
            state.fallback_reason = None
            state.fallback_time = None
            state.error_count = 0
    
    def reset_all(self) -> None:
        """Reset all agents to their primary models."""
        for agent_name in list(self._session_states.keys()):
            self.reset_to_primary(agent_name)
    
    def _get_or_create_state(self, agent_name: str) -> ModelState:
        """Get or create state for an agent."""
        agent_key = agent_name.lower()
        if agent_key not in self._session_states:
            agent_config = self._get_agent_config(agent_name)
            self._session_states[agent_key] = ModelState(
                current_model=agent_config.primary_model
            )
        return self._session_states[agent_key]
    
    def get_all_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration info for all defined agents."""
        return {
            name: {
                "primary": config.primary_model,
                "fallback": config.fallback_model,
                "description": config.description,
            }
            for name, config in self._agent_configs.items()
        }


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_model_manager: Optional[DynamicModelManager] = None


def get_model_manager() -> DynamicModelManager:
    """Get the global model manager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = DynamicModelManager()
    return _model_manager


def get_dynamic_model(agent_name: str) -> str:
    """
    Convenience function to get the current model for an agent.
    
    Usage in agent files:
        from utils.model_manager import get_dynamic_model
        
        # In Ketin's agent file:
        MODEL = get_dynamic_model("ketin")  # Returns Claude Opus 4.5
        
        # In Lira's agent file:
        MODEL = get_dynamic_model("lira")   # Returns Claude Sonnet 4.5
    """
    return get_model_manager().get_current_model(agent_name)


def get_model_info(agent_name: str) -> Dict[str, Any]:
    """
    Get detailed model info for display in UI.
    
    Usage in Streamlit:
        info = get_model_info("ketin")
        st.caption(f"Powered by {info['display_name']}")
    """
    return get_model_manager().get_model_display_info(agent_name)
