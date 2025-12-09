# Causal Impact: Glossary of Terms

## Core Statistical Terms

### Bayesian Structural Time Series (BSTS)
A statistical model that combines structural time series components (trend, seasonality) with Bayesian inference. BSTS models are the foundation of CausalImpact, allowing for uncertainty quantification and incorporation of prior knowledge.

### Counterfactual
The hypothetical outcome that would have occurred if the intervention had NOT taken place. In causal impact analysis, this is estimated using control variables and pre-period patterns. The counterfactual cannot be directly observed—it must be inferred.

### Credible Interval (Bayesian Confidence Interval)
A range of values within which the true effect likely falls, with a specified probability (typically 95%). Unlike frequentist confidence intervals, Bayesian credible intervals have a direct probability interpretation: "There is a 95% probability the true effect is in this range."

### Kalman Filter
A mathematical algorithm used to estimate the state of a system from noisy observations. In causal impact, the Kalman Filter helps estimate the underlying time series components and make predictions.

### Posterior Distribution
The updated probability distribution of parameters after observing data. In causal impact, posterior distributions describe our uncertainty about the causal effect after analysis.

### Posterior Probability
The probability that a hypothesis is true given the observed data. In causal impact results, "posterior probability of causal effect" indicates confidence that a genuine effect exists.

### Prior Distribution
Initial beliefs about parameters before observing data. In causal impact, priors can be specified for components like trend variance. Setting `prior_level_sd=None` in Python lets the algorithm optimize this automatically.

### Synthetic Control
A weighted combination of control variables designed to approximate what the treatment series would have looked like without intervention. The synthetic control IS the counterfactual prediction.

---

## Causal Inference Terms

### Attribution
Assigning credit for an outcome to specific causes. Causal impact helps attribute changes in metrics to specific interventions.

### Causal Effect / Treatment Effect
The difference in outcomes attributable to an intervention. Calculated as: `Causal Effect = Observed Outcome - Counterfactual Outcome`

### Confounding Variable / Confounder
A variable that influences both the treatment and the outcome, potentially creating a spurious association. Example: Seasonality could affect both when you run campaigns and sales, making it hard to isolate campaign effects.

### Control Group
A group not exposed to the intervention, used as a baseline for comparison. In causal impact, control time series serve this role.

### Difference-in-Differences (DiD)
A quasi-experimental method comparing changes in outcomes between treatment and control groups before and after intervention. Causal impact extends this concept with more sophisticated modeling.

### Endogeneity
When an explanatory variable is correlated with the error term, often due to reverse causality or omitted variables. Endogeneity can bias causal estimates.

### Intervention
The action, event, or treatment whose causal effect you want to measure. Examples: campaign launch, price change, feature release.

### Omitted Variable Bias
Bias arising from failing to control for a variable that affects both treatment and outcome. A major threat to causal inference from observational data.

### Parallel Trends Assumption
The assumption that treatment and control groups would have followed similar trends in the absence of intervention. This assumption underlies the validity of causal impact estimates.

### Quasi-Experiment
A study that estimates causal effects without random assignment, using statistical methods to create comparison groups. Causal impact is a quasi-experimental method.

### Selection Bias
Systematic differences between groups being compared that affect outcomes. Can arise if intervention timing or targeting is based on expected performance.

### Spillover / Contamination
When the intervention affects the control group, violating the assumption of independence. Example: A marketing campaign in one region affecting consumer behavior in "control" regions.

### Treatment Group
The group exposed to the intervention. In causal impact, this is typically the market, time period, or metric that experienced the change.

---

## Time Series Terms

### Autocorrelation
The correlation of a time series with a lagged version of itself. Time series data typically exhibit autocorrelation, which BSTS models account for.

### Pre-Intervention Period / Pre-Period
The time before the intervention, used to train the model and establish baseline patterns. Longer pre-periods generally improve model accuracy.

### Post-Intervention Period / Post-Period
The time after the intervention, during which we measure the effect. The model predicts what would have happened, and we compare to what actually happened.

### Seasonality
Regular, predictable patterns that repeat over time (daily, weekly, monthly, yearly). Models should account for seasonality when present.

### Stationarity
A statistical property indicating that a time series has constant mean and variance over time. Many time series methods assume or require stationarity.

### Structural Break
A sudden change in the underlying process generating a time series. Can invalidate models trained on pre-break data.

### Trend
The long-term increase or decrease in data. BSTS models can incorporate trend components.

---

## Marketing Measurement Terms

### Absolute Effect
The causal effect measured in original units (e.g., "250 additional conversions"). Contrast with relative effect.

### Average Effect
The average per-period causal effect during the post-intervention period.

### Cumulative Effect
The total causal effect summed across the entire post-intervention period.

### Incrementality
The additional outcomes caused specifically by a marketing action, beyond what would have happened anyway. Causal impact measures incrementality.

### Lift
The percentage increase in a metric attributable to an intervention. Example: "12% lift in conversions" means 12% more conversions than would have occurred without the intervention.

### Media Mix Modeling (MMM)
A statistical approach for measuring the impact of various marketing inputs on sales or other outcomes. Causal impact can complement MMM for specific intervention analysis.

### Relative Effect
The causal effect expressed as a percentage of the counterfactual baseline. Example: "8% lift" means observed was 8% higher than predicted counterfactual.

### Return on Ad Spend (ROAS)
Revenue generated per dollar of advertising spend. Causal impact can help measure incremental ROAS.

---

## Python/Technical Terms

### CausalImpact Class
The main class in the pycausalimpact library used to run analyses.

```python
from causalimpact import CausalImpact
ci = CausalImpact(data, pre_period, post_period)
```

### DataFrame
A pandas data structure used to hold the input data for causal impact analysis. Should have a datetime index with response variable in the first column.

### prior_level_sd
A parameter in pycausalimpact controlling the prior standard deviation of the local level component. Recommendation: Set to `None` to let the model optimize.

### summary()
Method to get numerical results from a CausalImpact analysis.

### summary(output='report')
Method to get a natural language interpretation of results.

### plot()
Method to visualize causal impact results showing original data, pointwise effect, and cumulative effect.

---

## Quick Reference Card

| Abbreviation | Full Term |
|--------------|-----------|
| BSTS | Bayesian Structural Time Series |
| CI | Causal Impact (or Confidence/Credible Interval depending on context) |
| DiD | Difference-in-Differences |
| MMM | Media Mix Modeling |
| MAPE | Mean Absolute Percentage Error |
| MCMC | Markov Chain Monte Carlo |
| ROAS | Return on Ad Spend |
| RCT | Randomized Controlled Trial |
| SCM | Synthetic Control Method |
