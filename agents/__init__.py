"""
Marketing Measurement Agents Package

This package contains the multi-agent system for marketing test design and measurement.

Note: We use measurement_lead_agent as the main coordinator which has test_design_agent
and measurement_agent as sub-agents. The orchestration_agent workflows are kept separate
as an alternative approach but not imported here to avoid agent parent conflicts.
"""
from .measurement_lead import measurement_lead_agent, root_agent

# Export the main agent for use
# Note: test_design_agent and measurement_agent are sub-agents of measurement_lead
# and should not be used independently once measurement_lead is instantiated

# Alias for convenience
measurement_lead = measurement_lead_agent

__all__ = [
    "measurement_lead_agent",
    "measurement_lead",
    "root_agent",
]
