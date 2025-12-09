# Causal Impact: Python Implementation Guide

## Available Python Packages

There are several Python implementations of Google's CausalImpact algorithm:

### 1. pycausalimpact (Recommended for most cases)
```bash
pip install pycausalimpact
```
- Uses statsmodels with Kalman Filter approach
- Well-documented, actively maintained
- Good balance of features and usability

### 2. tfcausalimpact (TensorFlow-based)
```bash
pip install tfcausalimpact
pip install 'tensorflow-probability[tf]'
```
- Uses TensorFlow Probability
- Designed to produce results close to R package
- Better for complex models, more computationally intensive

### 3. causalimpactx (Alternative)
```bash
pip install causalimpactx
```
- Alternative when pycausalimpact has compatibility issues
- Good for newer Python versions

## Basic Implementation with pycausalimpact

### Complete Example

```python
import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

# ============================================
# STEP 1: Prepare Your Data
# ============================================

# Data should be a DataFrame with:
# - DatetimeIndex (dates)
# - First column: response variable (y) - what you're measuring
# - Remaining columns: control variables (X1, X2, ...) - unaffected by intervention

# Example: Creating sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

# Simulate control series
X1 = 100 + np.cumsum(np.random.randn(100) * 2)
X2 = 80 + np.cumsum(np.random.randn(100) * 1.5)

# Simulate response: correlated with controls + intervention effect after day 70
y = 0.8 * X1 + 0.5 * X2 + np.random.randn(100) * 5
y[70:] += 15  # Intervention effect of +15 units

# Create DataFrame
data = pd.DataFrame({
    'y': y,       # Response variable MUST be first column
    'X1': X1,     # Control variable 1
    'X2': X2      # Control variable 2
}, index=dates)

print("Data shape:", data.shape)
print(data.head())

# ============================================
# STEP 2: Define Pre and Post Periods
# ============================================

# Using index positions (integer-based)
pre_period = [0, 69]    # Days 0-69 (before intervention)
post_period = [70, 99]  # Days 70-99 (after intervention)

# Alternative: Using dates (if using DatetimeIndex)
# pre_period = ['2023-01-01', '2023-03-11']
# post_period = ['2023-03-12', '2023-04-10']

# ============================================
# STEP 3: Run Causal Impact Analysis
# ============================================

# RECOMMENDED: Let model optimize the prior
ci = CausalImpact(
    data, 
    pre_period, 
    post_period,
    prior_level_sd=None  # Important: let statsmodels optimize
)

# ============================================
# STEP 4: View Results
# ============================================

# Numeric summary
print(ci.summary())

# Detailed report (natural language interpretation)
print(ci.summary(output='report'))

# ============================================
# STEP 5: Visualize Results
# ============================================

# Plot the analysis
ci.plot()
plt.tight_layout()
plt.savefig('causal_impact_results.png', dpi=150)
plt.show()
```

### Understanding the Plot Output

The `ci.plot()` function generates three panels:

1. **Original**: Actual data (solid) vs. counterfactual prediction (dashed)
2. **Pointwise**: Difference between observed and predicted at each time point
3. **Cumulative**: Running total of the causal effect over time

### Understanding the Summary Output

```
Posterior Inference {Causal Impact}
                          Average            Cumulative
Actual                    125.23             3756.86
Prediction (s.d.)         120.34 (0.31)      3610.28 (9.28)
95% CI                    [119.76, 120.97]   [3592.67, 3629.06]

Absolute effect (s.d.)    4.89 (0.31)        146.58 (9.28)
95% CI                    [4.26, 5.47]       [127.8, 164.19]

Relative effect (s.d.)    4.06% (0.26%)      4.06% (0.26%)
95% CI                    [3.54%, 4.55%]     [3.54%, 4.55%]

Posterior tail-area probability p: 0.001
Posterior prob. of a causal effect: 99.9%
```

**Key Metrics:**
- **Actual**: What was observed in post-period
- **Prediction**: Counterfactual (what would have happened)
- **Absolute effect**: Difference in original units
- **Relative effect**: Percentage change
- **Posterior probability**: Confidence that effect is real (>95% typically considered significant)

## Working with Real Data

### Loading Marketing Data

```python
import pandas as pd

# From CSV
data = pd.read_csv('marketing_data.csv', 
                   parse_dates=['date'], 
                   index_col='date')

# Ensure response variable is first column
data = data[['conversions', 'organic_traffic', 'competitor_spend', 'search_trends']]

# Check for missing values
print(data.isnull().sum())

# Fill or interpolate if needed
data = data.interpolate(method='time')
```

### Handling Different Time Granularities

```python
# Daily data
data_daily = data.resample('D').sum()

# Weekly data (often better for noisy data)
data_weekly = data.resample('W').sum()

# Monthly data
data_monthly = data.resample('M').sum()
```

## Validation Functions

### Placebo Test Implementation

```python
def run_placebo_test(data, true_intervention_idx, num_placebo=5):
    """
    Run placebo tests at points before the true intervention.
    If model is reliable, these should show no significant effect.
    """
    results = []
    
    # Generate placebo intervention points
    placebo_points = np.linspace(
        len(data) // 4,  # Start 25% into data
        true_intervention_idx - 10,  # End before true intervention
        num_placebo
    ).astype(int)
    
    for placebo_idx in placebo_points:
        pre_period = [0, placebo_idx - 1]
        post_period = [placebo_idx, true_intervention_idx - 1]
        
        try:
            ci = CausalImpact(data, pre_period, post_period, prior_level_sd=None)
            summary = ci.summary_data
            
            results.append({
                'placebo_idx': placebo_idx,
                'effect': summary['average']['abs_effect'],
                'p_value': summary['p_value'],
                'significant': summary['p_value'] < 0.05
            })
        except Exception as e:
            print(f"Placebo at {placebo_idx} failed: {e}")
    
    return pd.DataFrame(results)

# Run placebo tests
placebo_results = run_placebo_test(data, true_intervention_idx=70)
print(placebo_results)

# If many placebo tests are significant, be skeptical of real results
```

### Pre-Period Fit Assessment

```python
def assess_preperiod_fit(ci):
    """
    Assess how well the model fits the pre-period data.
    """
    # Get pre-period predictions
    pre_data = ci.inferences.loc[ci.pre_period[0]:ci.pre_period[1]]
    
    # Calculate metrics
    actual = pre_data['response']
    predicted = pre_data['point_pred']
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    correlation = np.corrcoef(actual, predicted)[0, 1]
    
    print(f"Pre-period Fit Assessment:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Correlation: {correlation:.4f}")
    
    # Rule of thumb thresholds
    if mape > 10:
        print("  WARNING: MAPE > 10% suggests poor pre-period fit")
    if correlation < 0.9:
        print("  WARNING: Low correlation suggests controls may not predict well")
    
    return {'mape': mape, 'rmse': rmse, 'correlation': correlation}
```

## Integration with Marketing Measurement Workflow

### Example: Campaign Impact Analysis

```python
class CampaignImpactAnalyzer:
    """
    Wrapper for analyzing marketing campaign impact using Causal Impact.
    """
    
    def __init__(self, response_col, control_cols, date_col='date'):
        self.response_col = response_col
        self.control_cols = control_cols
        self.date_col = date_col
        self.results = None
        
    def prepare_data(self, df, intervention_date):
        """Prepare data for analysis."""
        # Set date index
        data = df.set_index(self.date_col).copy()
        
        # Order columns: response first, then controls
        cols = [self.response_col] + self.control_cols
        data = data[cols]
        
        # Determine periods
        self.intervention_date = pd.to_datetime(intervention_date)
        self.pre_period = [data.index[0], self.intervention_date - pd.Timedelta(days=1)]
        self.post_period = [self.intervention_date, data.index[-1]]
        
        return data
    
    def analyze(self, df, intervention_date):
        """Run the causal impact analysis."""
        data = self.prepare_data(df, intervention_date)
        
        self.ci = CausalImpact(
            data, 
            self.pre_period, 
            self.post_period,
            prior_level_sd=None
        )
        
        self.results = {
            'summary': self.ci.summary(),
            'report': self.ci.summary(output='report'),
            'summary_data': self.ci.summary_data
        }
        
        return self.results
    
    def get_key_metrics(self):
        """Extract key metrics for reporting."""
        if self.results is None:
            raise ValueError("Run analyze() first")
            
        sd = self.ci.summary_data
        return {
            'average_effect': sd['average']['abs_effect'],
            'cumulative_effect': sd['cumulative']['abs_effect'],
            'relative_effect_pct': sd['average']['rel_effect'] * 100,
            'confidence_interval': (
                sd['average']['abs_effect_lower'],
                sd['average']['abs_effect_upper']
            ),
            'posterior_probability': 1 - sd['p_value'],
            'is_significant': sd['p_value'] < 0.05
        }

# Usage
analyzer = CampaignImpactAnalyzer(
    response_col='conversions',
    control_cols=['organic_traffic', 'competitor_sales', 'search_index']
)

results = analyzer.analyze(marketing_df, intervention_date='2023-09-01')
metrics = analyzer.get_key_metrics()

print(f"Campaign Effect: {metrics['relative_effect_pct']:.1f}%")
print(f"Probability of Real Effect: {metrics['posterior_probability']*100:.1f}%")
```

## Troubleshooting Common Issues

### Issue: "Model did not converge"
**Solution**: Try increasing iterations or simplifying model

### Issue: Poor pre-period fit
**Solution**: Add more/better control variables, check for data issues

### Issue: Implausibly large effects
**Solution**: Check assumptions, look for confounding events

### Issue: Wide confidence intervals
**Solution**: More data, better controls, or accept uncertainty
