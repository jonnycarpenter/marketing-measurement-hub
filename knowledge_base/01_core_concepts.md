# Causal Impact: Core Concepts

## What is Causal Impact?

Causal Impact is an algorithm developed by Google that uses **Bayesian Structural Time Series (BSTS)** models to estimate the causal effect of an intervention on a time series. It answers the fundamental question: "What would have happened if the intervention had not occurred?"

The methodology was published by Kay H. Brodersen et al. in the Annals of Applied Statistics (2015) and is available as open-source packages in both R and Python.

## The Fundamental Problem

In marketing and business analytics, we constantly face a challenge: we can observe what happened after we took an action (ran a campaign, changed pricing, launched a feature), but we cannot directly observe what *would have happened* if we hadn't taken that action. This unobserved scenario is called the **counterfactual**.

### Correlation vs. Causation

A critical distinction in causal analysis:

- **Correlation**: Two variables move together, but one doesn't necessarily cause the other
- **Causation**: One variable directly produces a change in another

Example of misleading correlation: Ice cream sales and sunglasses sales both increase in summer—but buying sunglasses doesn't cause people to buy ice cream. The shared cause is warmer weather.

## How Causal Impact Works

### The Two-Period Framework

1. **Pre-intervention Period**: Historical data before your intervention
   - Used to train a model that learns the relationship between your target metric and control variables
   - Establishes the baseline behavior and patterns

2. **Post-intervention Period**: Data after your intervention
   - The trained model predicts what would have happened WITHOUT the intervention
   - This prediction is the **synthetic counterfactual**
   - The difference between actual and predicted values is the estimated causal effect

### The Counterfactual Approach

```
Causal Effect = Observed Outcome - Counterfactual (Predicted) Outcome
```

The model constructs a "synthetic control" using:
- Control time series (metrics not affected by the intervention)
- Historical patterns and trends
- Seasonality components
- Regression relationships between variables

### Bayesian Framework Benefits

1. **Uncertainty Quantification**: Provides probability distributions, not just point estimates
2. **Credible Intervals**: 95% intervals tell you the range of plausible effect sizes
3. **Posterior Probability**: Probability that the effect is genuine (not due to chance)
4. **Temporal Evolution**: See how the effect evolves over time

## Key Terminology

| Term | Definition |
|------|------------|
| **Intervention** | The action/event whose effect you want to measure |
| **Response Variable** | The outcome metric you're measuring (e.g., sales, clicks) |
| **Control Variables** | Time series NOT affected by the intervention (used to build counterfactual) |
| **Pre-period** | Time before intervention (training data) |
| **Post-period** | Time after intervention (evaluation period) |
| **Counterfactual** | Predicted outcome if intervention hadn't occurred |
| **Absolute Effect** | Raw difference between observed and predicted |
| **Relative Effect** | Percentage change attributable to intervention |
| **Posterior Probability** | Confidence that a genuine effect exists |

## When to Use Causal Impact

### Ideal Use Cases

- Marketing campaign effectiveness measurement
- Pricing change impact analysis
- Product launch effects
- Policy change evaluation
- Feature rollout analysis
- Geographic expansion impact

### When Causal Impact is Preferred Over A/B Testing

- **Impossible to randomize**: TV campaigns, PR events, regulatory changes
- **Ethical concerns**: Can't withhold treatment from some users
- **Retrospective analysis**: Event already happened
- **Market-level interventions**: Changes affect entire markets

## Key Insight

Causal Impact is essentially a sophisticated "what-if" analysis that uses statistical modeling to estimate what you cannot directly observe: the world where your intervention never happened. The quality of this estimate depends heavily on having good control variables that were NOT affected by your intervention.
