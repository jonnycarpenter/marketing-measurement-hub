# Causal Impact: Marketing Use Cases

## Overview

Causal Impact analysis is particularly powerful in marketing contexts where traditional A/B testing is impractical or impossible. This document covers common marketing use cases with specific guidance for each.

## Use Case 1: Advertising Campaign Effectiveness

### Scenario
You launched a TV advertising campaign in a specific market and want to measure its impact on sales.

### Setup
- **Response Variable**: Sales in the target market
- **Control Variables**: 
  - Sales in non-targeted markets
  - Sales of unrelated product categories
  - Industry/category benchmarks
- **Intervention Date**: Campaign launch date

### Key Considerations
- **Spillover**: TV campaigns can have national reach even when targeted locally
- **Delayed Effects**: Brand advertising may have lagged impact
- **Control Selection**: Markets should have similar demographics and trends

### Example Control Selection
```
Good Controls:
- Sales in geographically distant markets
- Non-competing product lines
- Syndicated industry data

Poor Controls:
- Sales in adjacent markets (spillover risk)
- Digital channel performance (may be affected)
- Competitor sales (your gain may be their loss)
```

---

## Use Case 2: Pricing Change Impact

### Scenario
You increased prices by 10% and want to understand the impact on revenue and unit sales.

### Setup
- **Response Variables**: 
  - Revenue (primary)
  - Unit sales (secondary)
- **Control Variables**:
  - Competitor pricing/sales
  - Category-level trends
  - Economic indicators
- **Intervention Date**: Price change effective date

### Key Considerations
- **Anticipation Effects**: Customers may stockpile before increase
- **Gradual Rollout**: Phased changes complicate analysis
- **Channel Differences**: Online vs. retail may respond differently

### Analysis Approach
```python
# Analyze revenue impact
revenue_ci = CausalImpact(revenue_data, pre_period, post_period)

# Separately analyze volume impact  
volume_ci = CausalImpact(volume_data, pre_period, post_period)

# Price elasticity insight:
# If revenue up and volume down → inelastic demand (good)
# If revenue down and volume down → may have overpriced
```

---

## Use Case 3: Promotion/Discount Effectiveness

### Scenario
You ran a 20% off promotion during Black Friday week and want to measure incremental sales.

### Setup
- **Response Variable**: Sales during promotion period
- **Control Variables**:
  - Prior year same period (if similar)
  - Non-promoted product categories
  - Category trends from industry data
- **Intervention Period**: Promotion dates

### Key Considerations
- **Pull-Forward Effect**: Sales may cannibalize future periods
- **Seasonality**: Black Friday has inherent lift; isolate promotion effect
- **Competitor Promotions**: Everyone promotes during this period

### Recommended Approach
1. Use year-over-year controls if available
2. Extend post-period to capture pull-forward
3. Consider % discount as "dose" for dose-response analysis

---

## Use Case 4: New Product/Feature Launch

### Scenario
You launched a new product feature and want to measure impact on user engagement.

### Setup
- **Response Variable**: Engagement metric (DAU, time spent, etc.)
- **Control Variables**:
  - Engagement from users without the feature
  - Non-affected feature usage
  - Industry engagement benchmarks
- **Intervention Date**: Feature rollout date

### Key Considerations
- **Staged Rollouts**: If feature rolled out gradually, analysis is complex
- **Novelty Effect**: Initial spike may not reflect long-term impact
- **User Segments**: Different users may respond differently

### Best Practice
```
Recommended Post-Period Length:
- 2-4 weeks for immediate effect assessment
- 8-12 weeks for sustained effect confirmation
- Consider novelty decay in interpretation
```

---

## Use Case 5: Channel Mix Changes

### Scenario
You shifted 20% of your budget from Display to Search advertising and want to measure overall impact.

### Setup
- **Response Variable**: Total conversions/revenue
- **Control Variables**:
  - Organic channel performance
  - Non-affected paid channels
  - Market/category trends
- **Intervention Date**: Budget reallocation date

### Key Considerations
- **Channel Interaction**: Channels may have synergistic or cannibalistic effects
- **Attribution Complexity**: Same conversion may be influenced by multiple channels
- **Ramp-Up Time**: New channel allocation may take time to optimize

### Analysis Framework
```
Total Effect = Direct Effect + Synergy Effects - Cannibalization

Analyze:
1. Total conversions (overall health check)
2. Channel-specific conversions (direct attribution)
3. Assisted conversions (interaction effects)
```

---

## Use Case 6: Bid Strategy Changes

### Scenario
You switched from manual CPC bidding to Target ROAS automated bidding on Google Ads.

### Setup
- **Response Variable**: ROAS, conversions, or revenue
- **Control Variables**:
  - Other campaigns not changed
  - Organic search performance
  - Industry benchmarks
- **Intervention Date**: Strategy switch date

### Key Considerations
- **Learning Period**: Automated strategies need 2-4 weeks to optimize
- **Budget Constraints**: May need to exclude budget-limited periods
- **Auction Dynamics**: Market competition affects results

### Timing Recommendations
```
Pre-period: At least 30 days of stable manual bidding
Learning period: Exclude first 14-21 days after switch
Post-period: At least 30 days after learning period
```

---

## Use Case 7: Geographic Expansion

### Scenario
You expanded into a new market/region and want to measure the impact on overall business.

### Setup
- **Response Variable**: Total company revenue/growth
- **Control Variables**:
  - Performance in existing markets
  - Comparable companies without expansion
  - Market-specific economic indicators
- **Intervention Date**: Market launch date

### Key Considerations
- **Startup Costs**: Initial period may show losses
- **Growth Trajectory**: New markets have different growth curves
- **Existing Market Impact**: Expansion may affect existing operations

---

## Use Case 8: Competitive Response Analysis

### Scenario
A competitor launched a major campaign, and you want to understand its impact on your business.

### Setup
- **Response Variable**: Your sales/market share
- **Control Variables**:
  - Category-level trends
  - Other competitors' performance
  - Economic indicators
- **Intervention Date**: Competitor campaign launch

### Key Considerations
- **Information Asymmetry**: You may not know exact campaign details
- **Multiple Competitors**: Others may also be affected
- **Market Size**: Is it zero-sum or growing category?

---

## Selecting Control Variables by Use Case

| Use Case | Primary Controls | Secondary Controls |
|----------|-----------------|-------------------|
| Advertising | Non-targeted markets | Industry trends, competitors |
| Pricing | Competitor prices | Economic indicators |
| Promotions | Non-promoted products | Prior year data |
| Feature Launch | Non-affected users | Platform benchmarks |
| Channel Mix | Organic performance | Industry benchmarks |
| Bid Strategy | Unchanged campaigns | Market auction data |
| Geographic | Existing markets | Comparable businesses |
| Competitive | Category trends | Other competitors |

## Common Pitfalls by Use Case

### Advertising
❌ Using digital metrics as controls (often affected by TV)
❌ Ignoring brand search lift
❌ Too short post-period for brand effects

### Pricing
❌ Not accounting for stockpiling
❌ Ignoring competitive response
❌ Mixing multiple price changes

### Promotions
❌ Not measuring post-promotion dip
❌ Using promoted products as controls
❌ Ignoring inventory effects

### Feature Launch
❌ Not accounting for novelty decay
❌ Using affected users as controls
❌ Too short evaluation period

## Integration with MMM (Media Mix Modeling)

Causal Impact can complement MMM:

1. **Validate MMM coefficients**: Use CI for specific interventions to check MMM estimates
2. **Fill gaps**: CI for events not in MMM training data
3. **Quick reads**: CI for rapid campaign assessment while MMM updates quarterly
4. **Calibration**: Use CI results to calibrate MMM priors

```
MMM: Holistic, ongoing attribution across all channels
CI: Specific intervention impact with uncertainty quantification

Best Practice: Use both, cross-validate results
```
