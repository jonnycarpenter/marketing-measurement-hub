"""
Causal Impact analysis utilities for measuring marketing test results.

Uses tfp-causalimpact (Google's TensorFlow Probability-based CausalImpact) for robust
Bayesian structural time series analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CausalImpactAnalyzer:
    """
    Wrapper for Google's CausalImpact methodology for marketing measurement.
    
    Uses tfp-causalimpact (TensorFlow Probability-based) for robust Bayesian structural
    time series analysis. No fallback - CausalImpact or nothing.
    """
    
    def __init__(self):
        self._causalimpact_available = False
        
        # Import tfp-causalimpact (Google's TensorFlow Probability implementation)
        try:
            from causalimpact.causalimpact_lib import fit_causalimpact
            self._causalimpact_available = True
            self.fit_causalimpact = fit_causalimpact
            logger.info("Using tfp-causalimpact (TensorFlow Probability-based CausalImpact)")
        except ImportError:
            logger.error(
                "tfp-causalimpact package not available. Install with: pip install tfp-causalimpact"
            )
            raise ImportError("CausalImpact package is required. Install tfp-causalimpact.")
    
    def _detect_column(self, df: pd.DataFrame, candidates: List[str], description: str) -> str:
        """Auto-detect a column from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                logger.info(f"Auto-detected {description} column: {col}")
                return col
        raise KeyError(f"Could not find {description} column. Tried: {candidates}. Available: {df.columns.tolist()}")
    
    def prepare_time_series_data(
        self,
        sales_df: pd.DataFrame,
        test_customer_ids: List[str],
        control_customer_ids: List[str],
        date_column: str = None,
        value_column: str = None,
        customer_id_column: str = None,
        metric: str = "revenue",
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Prepare time series data for CausalImpact analysis.
        
        Auto-detects column names if not specified. Aggregates order-level data
        by date for test and control groups.
        
        Args:
            sales_df: Sales data with transactions (each row = one order)
            test_customer_ids: List of customer IDs in test group
            control_customer_ids: List of customer IDs in control group
            date_column: Column name for date (auto-detected if None)
            value_column: Column name for metric value (auto-detected if None)
            customer_id_column: Column name for customer ID (auto-detected if None)
            metric: What to measure - "revenue" (sum of order values) or "orders" (count of orders)
            freq: Frequency for time series (D=daily, W=weekly)
            
        Returns:
            DataFrame with test and control time series
        """
        # Auto-detect columns if not specified
        if date_column is None:
            date_column = self._detect_column(
                sales_df, 
                ["order_date", "transaction_date", "date", "purchase_date"],
                "date"
            )
        
        if customer_id_column is None:
            customer_id_column = self._detect_column(
                sales_df,
                ["customer_id", "cust_id", "customerid", "user_id"],
                "customer ID"
            )
        
        if value_column is None and metric == "revenue":
            value_column = self._detect_column(
                sales_df,
                ["order_value_usd", "order_total", "order_value", "revenue", "amount", "sales_amount", "total"],
                "value"
            )
        
        logger.info(f"Preparing time series: date={date_column}, customer={customer_id_column}, value={value_column}, metric={metric}")
        
        # Filter to test and control customers
        test_sales = sales_df[sales_df[customer_id_column].isin(test_customer_ids)].copy()
        control_sales = sales_df[sales_df[customer_id_column].isin(control_customer_ids)].copy()
        
        logger.info(f"Test group orders: {len(test_sales)}, Control group orders: {len(control_sales)}")
        
        # Convert dates
        test_sales[date_column] = pd.to_datetime(test_sales[date_column])
        control_sales[date_column] = pd.to_datetime(control_sales[date_column])
        
        # Aggregate by date based on metric type
        if metric == "orders":
            # Count orders (each row is one order)
            test_ts = test_sales.groupby(pd.Grouper(key=date_column, freq=freq)).size()
            control_ts = control_sales.groupby(pd.Grouper(key=date_column, freq=freq)).size()
            logger.info("Aggregating by ORDER COUNT (each row = 1 order)")
        else:
            # Sum revenue
            test_ts = test_sales.groupby(pd.Grouper(key=date_column, freq=freq))[value_column].sum()
            control_ts = control_sales.groupby(pd.Grouper(key=date_column, freq=freq))[value_column].sum()
            logger.info(f"Aggregating by REVENUE SUM ({value_column})")
        
        # Combine into single DataFrame
        result = pd.DataFrame({
            "test": test_ts,
            "control": control_ts
        })
        
        # Fill missing values
        result = result.fillna(0)
        
        logger.info(f"Time series prepared: {len(result)} periods, date range {result.index.min()} to {result.index.max()}")
        
        return result
    
    def run_causal_impact(
        self,
        data: pd.DataFrame,
        pre_period: Tuple[str, str],
        post_period: Tuple[str, str],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run CausalImpact analysis on prepared time series data.
        
        Args:
            data: DataFrame with 'test' and 'control' columns indexed by date
            pre_period: Tuple of (start_date, end_date) for pre-intervention period
            post_period: Tuple of (start_date, end_date) for post-intervention period
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with analysis results
        """
        # Early validation of period values
        if any(x is None for x in pre_period) or any(x is None for x in post_period):
            logger.error(f"Invalid period values - pre_period: {pre_period}, post_period: {post_period}")
            return {
                "success": False,
                "error": "Missing date values for pre_period or post_period. Cannot run analysis.",
                "lift_absolute": 0,
                "lift_percent": 0,
                "p_value": 1.0,
                "significant": False
            }
        
        if not self._causalimpact_available:
            return {
                "success": False,
                "error": "CausalImpact package not available. Install tfcausalimpact.",
                "lift_absolute": 0,
                "lift_percent": 0,
                "p_value": 1.0,
                "significant": False
            }
        
        return self._run_causalimpact_analysis(data, pre_period, post_period, alpha)
    
    def _run_causalimpact_analysis(
        self,
        data: pd.DataFrame,
        pre_period: Tuple[str, str],
        post_period: Tuple[str, str],
        alpha: float,
        _retry_count: int = 0
    ) -> Dict[str, Any]:
        """Run actual CausalImpact analysis using tfp-causalimpact."""
        # Validate period values are not None
        if any(x is None for x in pre_period) or any(x is None for x in post_period):
            return {
                "success": False,
                "error": "Missing date values for pre_period or post_period. Cannot run analysis.",
                "lift_absolute": 0,
                "lift_percent": 0,
                "p_value": 1.0,
                "significant": False
            }
        
        try:
            # Format data for tfp-causalimpact
            # tfp-causalimpact expects: first column = target (y), subsequent columns = covariates (X)
            ci_data = data[["test", "control"]].copy()
            
            # Check if pre-period is too long (can cause convergence issues)
            pre_start = pd.to_datetime(pre_period[0])
            pre_end = pd.to_datetime(pre_period[1])
            pre_days = (pre_end - pre_start).days
            
            # If pre-period > 180 days, shorten it to improve convergence
            MAX_PRE_DAYS = 180
            if pre_days > MAX_PRE_DAYS and _retry_count == 0:
                new_pre_start = (pre_end - pd.Timedelta(days=MAX_PRE_DAYS)).strftime('%Y-%m-%d')
                logger.warning(f"Pre-period too long ({pre_days} days). Shortening to {MAX_PRE_DAYS} days for better convergence.")
                pre_period = (new_pre_start, pre_period[1])
            
            # Run CausalImpact analysis using tfp-causalimpact
            logger.info(f"Running tfp-causalimpact with pre_period={pre_period}, post_period={post_period}")
            logger.info(f"Data shape: {ci_data.shape}, date range: {ci_data.index.min()} to {ci_data.index.max()}")
            
            # tfp-causalimpact uses fit_causalimpact() function
            result = self.fit_causalimpact(
                ci_data,
                pre_period=pre_period,
                post_period=post_period,
                alpha=alpha
            )
            
            # Check if result has valid data
            if result is None or result.summary is None or result.summary.empty:
                error_msg = "CausalImpact model fitting failed - no results returned."
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "lift_absolute": 0,
                    "lift_percent": 0,
                    "p_value": 1.0,
                    "significant": False
                }
            
            # Extract results from tfp-causalimpact summary DataFrame
            # Summary has 'average' and 'cumulative' rows with columns:
            # actual, predicted, predicted_lower, predicted_upper, abs_effect, rel_effect, p_value, etc.
            summary_df = result.summary
            cumulative = summary_df.loc['cumulative']
            
            actual = float(cumulative['actual'])
            predicted = float(cumulative['predicted'])
            lift_absolute = float(cumulative['abs_effect'])
            lift_percent = float(cumulative['rel_effect']) * 100  # Convert to percentage
            
            # Get confidence intervals for the effect
            ci_lower = float(cumulative['abs_effect_lower'])
            ci_upper = float(cumulative['abs_effect_upper'])
            
            # Get p-value directly from tfp-causalimpact
            p_value = float(cumulative['p_value'])
            
            # Generate summary string from the DataFrame
            summary_str = summary_df.to_string()
            
            # Generate a text report
            report = self._generate_report(
                actual=actual,
                predicted=predicted,
                lift_absolute=lift_absolute,
                lift_percent=lift_percent,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                pre_period=pre_period,
                post_period=post_period
            )
            
            # Calculate PRE-PERIOD model fit statistics from the series data
            # tfp-causalimpact series uses: 'observed', 'posterior_mean', etc.
            series_df = result.series
            pre_mask = (series_df.index >= pre_period[0]) & (series_df.index <= pre_period[1])
            pre_data = series_df[pre_mask]
            
            # Get actual values - tfp-causalimpact uses 'observed' column
            if "observed" in pre_data.columns:
                actual_col = "observed"
            elif "actual" in pre_data.columns:
                actual_col = "actual"
            else:
                actual_col = pre_data.columns[0]  # Fallback to first column
            
            # Get predicted values - tfp-causalimpact uses 'posterior_mean' column
            if "posterior_mean" in pre_data.columns:
                predicted_col = "posterior_mean"
            elif "predicted" in pre_data.columns:
                predicted_col = "predicted"
            else:
                predicted_col = pre_data.columns[1]  # Fallback to second column
            
            model_fit_stats = self._calculate_model_fit_stats(
                actual_values=pre_data[actual_col].values,
                predicted_values=pre_data[predicted_col].values
            )
            
            return {
                "success": True,
                "lift_absolute": lift_absolute,
                "lift_percent": lift_percent,
                "actual": actual,
                "predicted": predicted,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "significant": p_value < alpha,
                "confidence_level": int((1 - alpha) * 100),
                "summary": summary_str,
                "report": report,
                "pre_period": pre_period,
                "post_period": post_period,
                "inferences": series_df.to_dict() if series_df is not None else None,
                # Model fit statistics from pre-period
                "mape": model_fit_stats["mape"],
                "mae": model_fit_stats["mae"],
                "rmse": model_fit_stats["rmse"],
                "r_squared": model_fit_stats["r_squared"]
            }
            
        except Exception as e:
            logger.error(f"CausalImpact analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"CausalImpact analysis failed: {str(e)}",
                "lift_absolute": 0,
                "lift_percent": 0,
                "p_value": 1.0,
                "significant": False
            }
    
    def _estimate_p_value(
        self,
        effect: float,
        ci_lower: float,
        ci_upper: float
    ) -> float:
        """Estimate p-value from confidence interval."""
        # If CI doesn't cross zero, effect is significant
        if ci_lower > 0 or ci_upper < 0:
            # Rough approximation based on distance from zero
            se = (ci_upper - ci_lower) / (2 * 1.96)  # Assuming 95% CI
            if se > 0:
                z = abs(effect) / se
                from scipy import stats
                return 2 * (1 - stats.norm.cdf(z))
            return 0.01
        else:
            # CI crosses zero
            return 0.5
    
    def _generate_report(
        self,
        actual: float,
        predicted: float,
        lift_absolute: float,
        lift_percent: float,
        p_value: float,
        ci_lower: float,
        ci_upper: float,
        pre_period: Tuple[str, str],
        post_period: Tuple[str, str]
    ) -> str:
        """Generate a human-readable report from CausalImpact results."""
        significant = p_value < 0.05
        direction = "increase" if lift_absolute > 0 else "decrease"
        
        report = f"""
CausalImpact Analysis Report
============================

Analysis Period:
- Pre-period: {pre_period[0]} to {pre_period[1]}
- Post-period (treatment): {post_period[0]} to {post_period[1]}

Results:
- Actual (observed): ${actual:,.0f}
- Predicted (counterfactual): ${predicted:,.0f}
- Absolute effect: ${lift_absolute:,.0f}
- Relative effect: {lift_percent:.1f}%
- 95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]
- p-value: {p_value:.4f}

Interpretation:
During the post-intervention period, the response variable had an average value of 
${actual:,.0f}. In the absence of the intervention, we would have expected an average 
response of ${predicted:,.0f}. The intervention resulted in a {direction} of ${abs(lift_absolute):,.0f} 
({abs(lift_percent):.1f}%).

{"This effect is STATISTICALLY SIGNIFICANT (p < 0.05)." if significant else "This effect is NOT statistically significant (p >= 0.05)."}
{"We can conclude with 95% confidence that the intervention had a causal effect." if significant else "We cannot conclude that the intervention had a causal effect."}
"""
        return report.strip()
    
    def _calculate_model_fit_stats(
        self,
        actual_values: np.ndarray,
        predicted_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate model fit statistics from actual vs predicted values.
        
        These metrics assess how well the model fits the pre-period data,
        which is crucial for validating the counterfactual prediction.
        
        Args:
            actual_values: Array of actual/observed values (pre-period)
            predicted_values: Array of predicted values from the model
            
        Returns:
            Dictionary with MAPE, MAE, RMSE, and R-squared values
        """
        try:
            # Remove any NaN values
            mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
            actual = actual_values[mask]
            predicted = predicted_values[mask]
            
            if len(actual) == 0:
                logger.warning("No valid data points for model fit calculation")
                return {"mape": None, "mae": None, "rmse": None, "r_squared": None}
            
            # Mean Absolute Error (MAE)
            mae = np.mean(np.abs(actual - predicted))
            
            # Root Mean Squared Error (RMSE)
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # Mean Absolute Percentage Error (MAPE)
            # Avoid division by zero - only calculate where actual != 0
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            else:
                mape = None
            
            # R-squared (coefficient of determination)
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else None
            
            logger.info(f"Model fit stats - MAPE: {mape:.2f}%, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r_squared:.4f}")
            
            return {
                "mape": round(mape, 2) if mape is not None else None,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r_squared": round(r_squared, 4) if r_squared is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating model fit stats: {e}")
            return {"mape": None, "mae": None, "rmse": None, "r_squared": None}
    
    def _generate_summary(
        self,
        lift_absolute: float,
        lift_percent: float,
        actual: float,
        predicted: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float,
        alpha: float
    ) -> str:
        """Generate a human-readable summary of the analysis."""
        significant = p_value < alpha
        confidence = int((1 - alpha) * 100)
        
        direction = "increase" if lift_absolute > 0 else "decrease"
        
        summary = f"""
CAUSAL IMPACT ANALYSIS SUMMARY
==============================

Observed Data:
- Actual (Test): {actual:,.2f}
- Predicted (Counterfactual): {predicted:,.2f}

Impact:
- Absolute Effect: {lift_absolute:,.2f} ({direction})
- Relative Effect: {lift_percent:.2f}%
- {confidence}% Confidence Interval: [{ci_lower:,.2f}, {ci_upper:,.2f}]

Statistical Significance:
- p-value: {p_value:.4f}
- Significant at {confidence}% level: {'Yes' if significant else 'No'}

Interpretation:
"""
        
        if significant:
            summary += f"""
The intervention had a statistically significant {direction} of {abs(lift_percent):.2f}% 
on the target metric. We can be {confidence}% confident that the true effect 
lies between {ci_lower:,.2f} and {ci_upper:,.2f}.
"""
        else:
            summary += f"""
The results are not statistically significant at the {confidence}% confidence level.
We cannot conclusively attribute the observed {direction} of {abs(lift_percent):.2f}% 
to the intervention.
"""
        
        return summary
    
    def create_visualization_data(
        self,
        data: pd.DataFrame,
        pre_period: Tuple[str, str],
        post_period: Tuple[str, str],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for visualization of the causal impact analysis.
        
        Args:
            data: Original time series data
            pre_period: Pre-intervention period
            post_period: Post-intervention period
            results: Results from run_causal_impact
            
        Returns:
            Dictionary with visualization data
        """
        pre_start = pd.to_datetime(pre_period[0])
        pre_end = pd.to_datetime(pre_period[1])
        post_start = pd.to_datetime(post_period[0])
        post_end = pd.to_datetime(post_period[1])
        
        viz_data = {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "actual": data["test"].tolist(),
            "control": data["control"].tolist(),
            "pre_period": {
                "start": pre_period[0],
                "end": pre_period[1]
            },
            "post_period": {
                "start": post_period[0],
                "end": post_period[1]
            }
        }
        
        # Calculate predicted counterfactual
        pre_mask = (data.index >= pre_start) & (data.index <= pre_end)
        pre_ratio = data[pre_mask]["test"].sum() / data[pre_mask]["control"].sum() if data[pre_mask]["control"].sum() > 0 else 1
        
        predicted = (data["control"] * pre_ratio).tolist()
        viz_data["predicted"] = predicted
        
        # Mark intervention point
        viz_data["intervention_date"] = post_period[0]
        
        return viz_data


# Singleton instance
causal_impact_analyzer = CausalImpactAnalyzer()
