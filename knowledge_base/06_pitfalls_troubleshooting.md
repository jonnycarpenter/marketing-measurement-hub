# Causal Impact: Common Pitfalls and Troubleshooting

## Overview

Even with a sophisticated tool like Causal Impact, there are many ways analyses can go wrong. This document catalogs common mistakes, how to identify them, and how to fix them.

---

## Pitfall 1: Contaminated Control Variables

### The Problem
Your control variables were actually affected by the intervention, biasing the counterfactual.

### Signs of This Problem
- Control variables show unusual patterns at intervention time
- Results seem too good (or too bad) to be true
- Domain experts are skeptical of chosen controls

### Example
Using "organic search traffic" as a control for a TV campaign, but the TV ads drive branded searches that boost organic traffic.

### How to Fix
1. Carefully reason through causal pathways before analysis
2. Plot controls and visually inspect for changes at intervention
3. Consult domain experts about potential contamination
4. Use multiple independent control sources
5. Test sensitivity to different control sets

---

## Pitfall 2: Insufficient Pre-Period Data

### The Problem
Not enough historical data to establish stable patterns and relationships.

### Signs of This Problem
- Wide confidence intervals
- Poor pre-period model fit
- Unstable results with slight changes

### Minimum Requirements
| Data Frequency | Minimum Pre-Period | Maximum Pre-Period |
|---------------|-------------------|-------------------|
| Daily | 60-90 days | 180 days |
| Weekly | 12-16 weeks | 26 weeks |
| Monthly | 12-24 months | 24 months |

**Note**: Pre-periods longer than 180 days (daily) can cause model convergence issues with Bayesian structural time series. The system automatically shortens overly long pre-periods to ensure reliable results.

### How to Fix
1. Use 90-180 days of pre-period for daily data (optimal range)
2. Aggregate to coarser granularity (daily → weekly) for longer time spans
3. Add more informative control variables
4. Accept wider uncertainty intervals

---

## Pitfall 3: Confounding Events

### The Problem
Other events occurred near the intervention time that could explain the observed effect.

### Signs of This Problem
- Unusually large or small effects
- Effect timing doesn't match intervention
- Known external events coincide with analysis period

### Common Confounders
- Holidays and seasonal events
- Competitor actions
- Economic news
- Product changes
- PR events
- Weather events

### How to Fix
1. Create an "event calendar" before analysis
2. Check news for relevant events around intervention
3. Extend analysis to include event effects
4. Acknowledge confounders in interpretation
5. Consider robustness checks excluding confounded periods

---

## Pitfall 4: Multiple Overlapping Interventions

### The Problem
Several interventions occur close together, making it impossible to isolate individual effects.

### Signs of This Problem
- Multiple marketing activities in analysis window
- Unclear which intervention to attribute effects to
- Results change depending on intervention date choice

### Example
Running email campaign, paid search changes, and website redesign all within 2 weeks.

### How to Fix
1. Plan interventions with measurement in mind (stagger timing)
2. Analyze combined effect and accept attribution uncertainty
3. Use longer post-periods to let effects separate
4. Consider MMM instead for multi-intervention scenarios

---

## Pitfall 5: Misspecified Time Periods

### The Problem
Pre and post periods don't correctly capture the intervention effect.

### Common Mistakes
- Pre-period too short (unstable baseline)
- Pre-period too long (relationship changes)
- Post-period too short (effect not yet visible)
- Wrong intervention date (early announcement effects, delayed implementation)

### Guidelines
```
Pre-period: Long enough to capture seasonality + stable relationships
Post-period: Long enough for effect to materialize + stabilize
Intervention: When the change actually reached customers
```

### How to Fix
1. Clearly define when customers experienced the change
2. Account for gradual rollouts
3. Consider anticipation effects (pre-announce effects)
4. Run sensitivity analysis with different period definitions

---

## Pitfall 6: Poor Model Fit

### The Problem
The model doesn't adequately explain the pre-period data, making counterfactual predictions unreliable.

### Signs of This Problem
- MAPE > 10% in pre-period
- Pre-period actuals frequently outside confidence intervals
- Correlation < 0.9 between predicted and actual
- Systematic over/under-prediction

### Diagnostic Code
```python
# Check pre-period fit
pre_period_data = ci.inferences.loc[:intervention_date]
actual = pre_period_data['response']
predicted = pre_period_data['point_pred']

mape = np.mean(np.abs((actual - predicted) / actual)) * 100
print(f"Pre-period MAPE: {mape:.1f}%")  # Want < 10%

# Visual check
plt.figure(figsize=(12, 4))
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.fill_between(pre_period_data.index, 
                 pre_period_data['pred_lower'],
                 pre_period_data['pred_upper'],
                 alpha=0.2)
plt.legend()
plt.title('Pre-Period Fit Check')
plt.show()
```

### How to Fix
1. Add more/better control variables
2. Check for structural breaks in pre-period
3. Try different time aggregations
4. Examine data quality issues
5. Consider if causal impact is appropriate for this problem

---

## Pitfall 7: Cherry-Picking Results

### The Problem
Running multiple analyses and reporting only favorable results.

### Signs of This Problem
- Trying different control sets until finding "significant" results
- Adjusting periods until effect appears
- Not pre-registering analysis plan

### How to Fix
1. Document analysis plan BEFORE looking at results
2. Pre-specify controls, periods, and success criteria
3. Report all analyses run, not just favorable ones
4. Use placebo tests regardless of main results
5. Have independent reviewer validate approach

---

## Pitfall 8: Ignoring Uncertainty

### The Problem
Reporting point estimates without acknowledging uncertainty, leading to overconfidence.

### Signs of This Problem
- Presenting effects as exact numbers
- Not reporting confidence intervals
- Making decisions as if estimates were certain

### How to Fix
Always report:
- Point estimate with confidence interval
- Posterior probability of effect
- Key assumptions and limitations
- Sensitivity of results to choices

### Example Good Reporting
```
"The campaign generated an estimated lift of 8.3% 
(95% CI: 3.1% to 13.5%) with 97% probability of a 
positive effect. This analysis assumes organic traffic 
was unaffected by the campaign."
```

---

## Pitfall 9: Causation Overclaims

### The Problem
Claiming causal impact proved causation when analysis only provides suggestive evidence.

### What Causal Impact Does
- Estimates what WOULD have happened
- Quantifies uncertainty in that estimate
- Provides probability of non-zero effect

### What It Doesn't Do
- PROVE causation
- Rule out all confounders
- Validate its own assumptions

### Better Language
| Don't Say | Do Say |
|-----------|--------|
| "Campaign caused 10% lift" | "Estimated 10% lift attributed to campaign" |
| "Analysis proves effectiveness" | "Analysis suggests positive effect" |
| "Definitive ROI" | "Estimated ROI under stated assumptions" |

---

## Pitfall 10: Technical Issues

### Data Problems
| Issue | Sign | Fix |
|-------|------|-----|
| Missing data | Gaps in time series | Interpolate or exclude |
| Outliers | Extreme values | Investigate and potentially winsorize |
| Unit mismatch | Scales differ | Standardize or log-transform |
| Timezone issues | Off-by-one errors | Align all data to same timezone |

### Model Problems
| Issue | Sign | Fix |
|-------|------|-----|
| Non-convergence | Error messages | Simplify model, more iterations |
| Numerical instability | Extreme estimates | Scale data, check for constants |
| Seasonality ignored | Periodic errors | Add seasonal controls |

---

## Troubleshooting Decision Tree

```
Is pre-period fit good?
├── No → Add controls, check data quality, extend pre-period
└── Yes → 
    Are placebo tests clean?
    ├── No → Model may be overfitting or controls contaminated
    └── Yes → 
        Are results plausible?
        ├── No → Check for confounders, validate assumptions
        └── Yes → 
            Is uncertainty acceptable?
            ├── No → Add data/controls, accept wider bounds
            └── Yes → Proceed with appropriate caveats
```

## Quick Reference: Red Flags

🚩 Effects > 50% (implausibly large)
🚩 MAPE > 10% in pre-period
🚩 Confidence intervals include zero AND large effects
🚩 Placebo tests show significant effects
🚩 Results change dramatically with small changes
🚩 Known confounding events unaddressed
🚩 Control variables show changes at intervention
🚩 Pre-period < 3x post-period length
🚩 Multiple overlapping interventions
🚩 No pre-registration of analysis plan
