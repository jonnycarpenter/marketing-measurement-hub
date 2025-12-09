# Agent Design

This document details the design and behavior of each AI agent in the system.

## Agent Overview

The system uses three specialized agents orchestrated via Google ADK:

| Agent | File | Primary Model |
|-------|------|---------------|
| Lira Maven | `measurement_lead.py` | Gemini 2.5 Pro |
| Ketin Vale | `test_design_agent.py` | Gemini 2.5 Flash |
| Brena Colette | `measurement_agent.py` | Gemini 2.5 Flash |

## Lira Maven (Lead Agent)

**Role:** Measurement Lead / Orchestrator

**Responsibilities:**
- Greet users and explain hub capabilities
- Route requests to appropriate specialist
- Maintain conversation context across handoffs
- Handle general questions about marketing measurement

**Prompt Highlights:**
```
You are Lira Maven, Measurement Lead at Outdoorsy Living.
You guide users through the Marketing Measurement Hub.
Route to Ketin for test design, Brena for measurement.
```

**Handoff Logic:**
- Test design keywords → Ketin
- Measurement/results keywords → Brena
- General questions → Handle directly

## Ketin Vale (Test Design Specialist)

**Role:** Experiment Design Expert

**Responsibilities:**
- Design statistically valid marketing experiments
- Select appropriate DMAs for geo-tests
- Check for conflicts with existing tests
- Validate audience splits for balance
- Save test configurations

**Key Tools Used:**
- `query_data` - Check promo calendar, existing tests
- `search_knowledge_base` - Retrieve best practices
- `save_test_to_master` - Persist approved designs
- `validate_audience_split` - Check test/control balance

**Prompt Highlights:**
```
You are Ketin Vale, Test Design Specialist.
CRITICAL: Stop after asking questions. Never assume user responses.
Always check for conflicts before finalizing designs.
```

**Design Process:**
1. Gather test objective from user
2. Query promo calendar for conflicts
3. Query existing tests for overlap
4. Recommend DMAs based on criteria
5. Present design for approval
6. Save to master testing doc

## Brena Colette (Measurement Analyst)

**Role:** Causal Inference Expert

**Responsibilities:**
- Run CausalImpact analysis on completed tests
- Interpret statistical results
- Generate executive summaries
- Provide go/no-go recommendations

**Key Tools Used:**
- `query_data` - Retrieve test data
- `run_causal_impact` - Execute measurement
- `search_knowledge_base` - Explain methodology
- `create_chart` - Visualize results

**Prompt Highlights:**
```
You are Brena Colette, Measurement Analyst.
Explain results in business terms, not just statistics.
Always provide confidence intervals and caveats.
```

**Analysis Process:**
1. Load test configuration
2. Prepare time series data
3. Run CausalImpact model
4. Generate diagnostics
5. Interpret results in business context
6. Provide recommendation

## Agent Communication

### Handoff Protocol

Agents hand off using ADK's built-in transfer mechanism:

```python
# In Lira's prompt
"To hand off to Ketin, use: transfer_to_ketin"
"To hand off to Brena, use: transfer_to_brena"
```

### Context Preservation

Session state maintains context across handoffs:
- User's original request
- Intermediate results
- Current test being discussed
- Conversation history

### Tool Sharing

All agents share the same tool definitions but use them differently:
- Ketin primarily uses `query_data` for conflict checking
- Brena primarily uses `run_causal_impact` for measurement
- Both use `search_knowledge_base` for best practices

## Prompt Engineering Notes

### Preventing Self-Answering

Ketin had a tendency to answer his own questions. Fixed with explicit rules:

```
CRITICAL CONVERSATION RULES:
1. STOP AFTER QUESTIONS - When you ask a question, stop immediately
2. ONE TURN RULE - Each response does ONE action only

WRONG: "Does that sound correct? Great. The objective is set."
CORRECT: "Does that sound correct?" [STOP]
```

### Grounding in Data

Agents are instructed to always query data before making claims:

```
Before recommending DMAs, ALWAYS:
1. Query the promo_calendar for the test dates
2. Query master_testing_doc for existing tests
3. Check for geographic overlap
```

### Business Context

Agents translate statistical output to business language:

```
Instead of: "The posterior probability is 0.97"
Say: "We're 97% confident the campaign drove incremental lift"
```
