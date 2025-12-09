# Causal Impact Methodology

This document explains the causal inference methodology used for marketing measurement.

## What is CausalImpact?

CausalImpact is a Bayesian structural time series method developed by Google. It estimates the causal effect of an intervention (e.g., a marketing campaign) by comparing:

1. **What actually happened** (observed data)
2. **What would have happened** without the intervention (counterfactual)

## Why Bayesian?

Traditional A/B tests require randomization. But many marketing interventions can't be randomized:
- Geo-targeted campaigns
- Brand campaigns
- Seasonal promotions

CausalImpact uses a **synthetic control** approach:
- Build a model of the target metric using pre-period data
- Project what the metric "should" have been during the test
- The difference is the causal effect

## Key Concepts

### Pre-Period vs Post-Period

```
        Pre-Period              Post-Period
    (Model Training)         (Effect Estimation)
|----------------------|--------------------------|
                       ^
                 Intervention
                 (Campaign Start)
```

- **Pre-period**: Historical data before the intervention
- **Post-period**: Data during/after the intervention
- The model learns patterns in the pre-period and applies them to estimate the counterfactual

### Synthetic Control

The model uses control variables (covariates) to predict the target:

```
Revenue_test = f(Revenue_control, Seasonality, Trend)
```

For geo-tests:
- **Test DMAs**: Markets receiving the intervention
- **Control DMAs**: Similar markets without intervention
- Control markets help predict what test markets "would have done"

### Credible Intervals

Unlike frequentist confidence intervals, Bayesian credible intervals have an intuitive interpretation:

> "There is a 95% probability the true effect lies within this range"

## Interpreting Results

### Key Metrics

| Metric | Meaning |
|--------|---------|
| **Average Lift** | Daily incremental effect |
| **Cumulative Lift** | Total incremental effect over test period |
| **Relative Lift %** | Effect as percentage of baseline |
| **Prob(Lift > 0)** | Probability the effect is positive |
| **P-value Equivalent** | Frequentist-style significance |

### Example Interpretation

```
Cumulative Lift: $158,544
Credible Interval: [$59,768, $259,379]
Prob(Lift > 0): 99.9%
Relative Lift: 8.4%
```

**Business Translation:**
> "The YouTube budget increase drove an estimated $158K in incremental revenue, with 99.9% confidence that the true effect is positive. We're 95% confident the actual lift is between $60K and $259K."

## Best Practices

### Pre-Period Length

- **Minimum**: 3x the post-period length
- **Recommended**: 6-12 months for seasonal businesses
- **Why**: Need enough data to capture patterns

### Test Duration

- **Minimum**: 2-4 weeks
- **Recommended**: 4-8 weeks
- **Why**: Need time for effects to materialize and stabilize

### Control Selection

Good controls should be:
- **Correlated** with test units in the pre-period
- **Unaffected** by the intervention
- **Stable** (no major changes during test)

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Contamination | Control exposed to treatment | Geographic separation |
| Spillover | Test affects control indirectly | Larger buffer zones |
| Short pre-period | Unstable model | Extend pre-period |
| Conflicting tests | Multiple interventions | Check promo calendar |

## Implementation Details

### Data Preparation

```python
# Required columns
data = pd.DataFrame({
    'date': [...],           # Daily dates
    'y': [...],              # Target metric (test markets)
    'x1': [...],             # Control metric 1
    'x2': [...],             # Control metric 2 (optional)
})
```

### Model Fitting

```python
from causalimpact import CausalImpact

pre_period = ['2025-01-01', '2025-03-01']
post_period = ['2025-03-02', '2025-03-29']

ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
```

### Diagnostics

Always check:
1. **Pre-period fit**: Does the model track the target well?
2. **Residuals**: Are they normally distributed?
3. **Stationarity**: Is the time series stable?

## References

- [Google CausalImpact Paper](https://research.google/pubs/pub41854/)
- [tfp-causalimpact Documentation](https://github.com/google/tfp-causalimpact)
- [Brodersen et al. (2015)](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.full)
