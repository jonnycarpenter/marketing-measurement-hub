# Critical Assumptions for Valid Causal Impact Analysis

## Overview

Causal Impact analysis relies on strong assumptions. Violating these assumptions can lead to incorrect conclusions about the effectiveness of marketing interventions. Understanding and validating these assumptions is **critical** for obtaining valid results.

## The Three Core Assumptions

### 1. Control Variables Were NOT Affected by the Intervention

**This is the most critical assumption.**

The control time series used to build the counterfactual must be completely unaffected by your intervention. If they were affected, your counterfactual will be biased.

**Examples of Violation:**
- Using sales in Canada as a control for a US advertising campaign, but the campaign spills over to Canadian consumers
- Using organic search as a control for paid search changes, but paid search cannibalizes organic
- Using competitor sales as control, but your campaign steals their market share

**How to Validate:**
- Plot all control variables and visually inspect for changes at intervention time
- Use domain knowledge to reason through potential spillover effects
- Consider geographic, channel, and temporal contamination

### 2. Relationship Between Treatment and Control is Stable

The relationship established during the pre-period must remain stable during the post-period. The model assumes the patterns it learned will continue.

**Examples of Violation:**
- Seasonality patterns change (new competitor enters, economic shift)
- External shock affects treatment and control differently
- Structural market changes during post-period

**How to Validate:**
- Ensure pre-period is long enough to capture patterns (at least 3x expected effect duration)
- Check for external events that could differentially affect groups
- Run placebo tests at various pre-period points

### 3. No Confounding External Events

No other significant events should occur at or near the intervention time that could explain the observed effect.

**Examples of Violation:**
- Competitor goes out of business same week as your campaign
- Holiday or seasonal event coincides with intervention
- Economic news or policy change at same time

**How to Validate:**
- Document all known external events
- Check industry news around intervention date
- Consider running analysis with different intervention dates

## The Parallel Trends Assumption

Underlying these assumptions is the concept of **parallel trends**: absent the intervention, the treatment and control series would have continued to move together as they did in the pre-period.

```
Pre-period:  Treatment ≈ f(Control) + noise
Post-period: Treatment (without intervention) ≈ f(Control) + noise  [assumed]
```

## Assumption Validation Checklist

Before running analysis:

- [ ] **Control Selection**: Can I articulate WHY each control variable was unaffected?
- [ ] **Spillover Check**: Is there any mechanism by which my intervention could affect controls?
- [ ] **Temporal Stability**: Are there reasons to believe relationships changed?
- [ ] **External Events**: What else happened around the intervention date?
- [ ] **Pre-period Fit**: Does my model fit the pre-period data well?

After running analysis:

- [ ] **Placebo Tests**: Does running on fake intervention dates show no effect?
- [ ] **Sensitivity Analysis**: Do results change dramatically with different controls?
- [ ] **Visual Inspection**: Do the counterfactual predictions look reasonable?

## Common Assumption Violations in Marketing

### Cross-Channel Contamination
Changing one marketing channel often affects others. Increasing paid search might cannibalize organic; TV ads drive branded search; social campaigns create word-of-mouth.

### Geographic Spillover
National campaigns affect all regions. Even "local" interventions can have spillover through travel, media exposure, or word-of-mouth.

### Temporal Contamination
Effects may occur before official intervention date (anticipation effects) or after (delayed effects, learning curves).

### Selection Bias
If you chose WHEN to intervene based on expected outcomes, your timing is not random and estimates may be biased.

## Red Flags That Assumptions May Be Violated

1. **Control variables show unusual patterns** right at intervention time
2. **Pre-period fit is poor** - model can't predict treatment from controls
3. **Results are implausibly large** - effects seem too good to be true
4. **Multiple interventions overlap** - hard to attribute effects
5. **Very short pre-period** - insufficient data to establish patterns
6. **Highly volatile data** - too much noise to detect signals

## Key Takeaway

> "With great statistical power comes great responsibility for assumptions."

Causal Impact provides sophisticated machinery for causal inference, but **no statistical technique can overcome violated assumptions**. Always spend significant effort validating that your analysis setup meets the required assumptions before trusting the results.
