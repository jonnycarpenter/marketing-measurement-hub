# Causal Impact Best Practices

## Data Preparation Best Practices

### Pre-Period Length

**Recommendation: Pre-period should be at least 3x the length of the post-period**

- Minimum: Enough data to capture seasonal patterns (at least one full cycle)
- Ideal: Multiple seasonal cycles plus sufficient observations for model training
- Too short: Model can't learn stable relationships
- Too long: Relationships may have changed over time

**Example:**
- Analyzing a 30-day campaign? Use at least 90 days of pre-period data
- Analyzing a week-long promotion? Use at least 3-4 weeks of pre-period data

### Control Variable Selection

**The most critical decision in your analysis.**

**Good Control Variables:**
- Highly correlated with response variable in pre-period
- Clearly unaffected by the intervention
- Same granularity (daily, weekly) as response variable
- No missing data during analysis period

**Sources of Control Variables:**
- Non-affected geographic regions (geo controls)
- Non-affected product categories
- Non-affected marketing channels
- Industry benchmarks or indices
- Google Trends for related (but unaffected) topics
- Competitor data (if available and unaffected)

**Control Selection Process:**
1. List all potential controls
2. For each, explicitly reason through whether intervention could affect it
3. Check correlation with response in pre-period
4. Prefer multiple diverse controls over single control
5. Let the model perform automatic variable selection when possible

### Data Quality Requirements

- **No gaps**: Complete time series for all variables
- **Consistent measurement**: Same metrics throughout
- **Aligned timestamps**: All series on same time index
- **Sufficient variation**: Not flat or constant series
- **No outliers**: Or handle them explicitly before analysis

## Model Configuration Best Practices

### Python Implementation (pycausalimpact)

```python
from causalimpact import CausalImpact

# Recommended configuration
ci = CausalImpact(
    data,
    pre_period,
    post_period,
    prior_level_sd=None  # Let model optimize the prior
)
```

**Key Recommendation**: Set `prior_level_sd=None` to let statsmodels optimize the prior on the local level component. Using arbitrary values risks sub-optimal solutions.

### R Implementation

```r
impact <- CausalImpact(
    data, 
    pre.period, 
    post.period,
    model.args = list(
        niter = 5000,  # More iterations for stability
        nseasons = 7   # Weekly seasonality if daily data
    )
)
```

## Validation Best Practices

### 1. Pre-Period Fit Check

Before trusting post-period results, verify the model fits pre-period data well.

- Plot predicted vs actual for pre-period
- Check if confidence intervals contain actual values
- If pre-period fit is poor, results are unreliable

### 2. Placebo Tests (Pseudo-Interventions)

Run the analysis pretending the intervention happened at different points in the pre-period where NO intervention actually occurred.

```python
# Test at multiple placebo dates
placebo_dates = ['2023-06-01', '2023-07-01', '2023-08-01']

for date in placebo_dates:
    # Run analysis with fake intervention date
    # Should show NO significant effect
```

**Interpretation:**
- If placebo tests show significant effects: Model is unreliable
- If placebo tests show no effects: More confidence in real results

### 3. Sensitivity Analysis

Test how results change with different choices:
- Different control variables
- Different pre-period lengths
- Different model specifications

**Robust results** should be consistent across reasonable variations.

### 4. Visual Inspection

Always plot results and visually inspect:
- Does counterfactual look reasonable?
- Does the effect timing make sense?
- Are confidence intervals plausible?

## Interpretation Best Practices

### Reading the Summary Output

Key metrics to focus on:

| Metric | What It Tells You |
|--------|-------------------|
| **Average Absolute Effect** | Per-period impact in original units |
| **Cumulative Effect** | Total impact over entire post-period |
| **95% Credible Interval** | Range of plausible effect sizes |
| **Relative Effect (%)** | Percentage lift from intervention |
| **Posterior Probability** | Confidence that effect is real |

### Statistical Significance

- **p < 0.05** (or posterior prob > 95%): Conventionally "significant"
- **Large confidence intervals**: Uncertain about effect size even if significant
- **Effect crosses zero**: Cannot rule out no effect or opposite effect

### Communicating Results

**Do Say:**
- "The analysis suggests an increase of X units (95% CI: [a, b])"
- "There is a Y% probability that the intervention had a positive effect"
- "The relative effect was approximately Z%, though uncertainty remains"

**Don't Say:**
- "The campaign caused exactly X increase" (ignores uncertainty)
- "The analysis proves causation" (it estimates, doesn't prove)
- Results without acknowledging assumptions

## Common Pitfalls to Avoid

### 1. Cherry-Picking Controls
Don't select controls based on which give you the results you want. Document control selection rationale BEFORE running analysis.

### 2. Ignoring Model Fit
Poor pre-period fit means unreliable results. Always check before interpreting.

### 3. Over-Interpreting Precision
Just because CI is narrow doesn't mean the effect is real—it means IF the model is right, the effect is precisely estimated.

### 4. Ignoring Context
Statistical significance without business sense should prompt investigation, not celebration.

### 5. Single Analysis
One analysis is rarely definitive. Use multiple validation approaches.

### 6. Short Time Horizons
Some effects take time to materialize. Don't conclude "no effect" too quickly.

### 7. Ignoring Uncertainty
Always report confidence/credible intervals, not just point estimates.

## Documentation Best Practices

For every analysis, document:

1. **Business Question**: What are you trying to learn?
2. **Intervention Definition**: Exactly what changed and when?
3. **Data Sources**: Where did each variable come from?
4. **Control Rationale**: Why was each control selected?
5. **Assumption Validation**: How were assumptions checked?
6. **Model Specification**: What parameters were used?
7. **Validation Results**: Placebo tests, sensitivity analysis
8. **Limitations**: What could bias results?
9. **Conclusions**: What did you learn?
