"""
Test design utilities for creating test/control splits.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class TestDesignUtils:
    """Utilities for designing marketing test/control experiments."""
    
    @staticmethod
    def generate_test_id() -> str:
        """Generate a unique test ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"TEST_{timestamp}"
    
    @staticmethod
    def calculate_sample_size(
        baseline_conversion: float,
        expected_lift: float,
        confidence_level: float = 0.95,
        power: float = 0.80,
        two_sided: bool = True
    ) -> int:
        """
        Calculate required sample size for a conversion rate test.
        
        Args:
            baseline_conversion: Current conversion rate (e.g., 0.05 for 5%)
            expected_lift: Expected relative lift (e.g., 0.10 for 10% lift)
            confidence_level: Statistical confidence level (default 0.95)
            power: Statistical power (default 0.80)
            two_sided: Whether the test is two-sided (default True)
            
        Returns:
            Required sample size per group
        """
        p1 = baseline_conversion
        p2 = baseline_conversion * (1 + expected_lift)
        
        # Z-scores
        alpha = 1 - confidence_level
        if two_sided:
            alpha = alpha / 2
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Sample size formula
        numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        n = numerator / denominator
        return int(np.ceil(n))
    
    @staticmethod
    def random_split(
        customer_df: pd.DataFrame,
        test_ratio: float = 0.5,
        stratify_by: Optional[List[str]] = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a random split of customers into test and control groups.
        
        Args:
            customer_df: DataFrame with customer data
            test_ratio: Proportion of customers in test group
            stratify_by: List of columns to stratify by
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (test_df, control_df)
        """
        np.random.seed(random_state)
        
        if stratify_by is not None and len(stratify_by) > 0:
            # Stratified sampling
            test_indices = []
            
            # Group by stratification columns
            grouped = customer_df.groupby(stratify_by)
            
            for name, group in grouped:
                n_test = int(len(group) * test_ratio)
                group_test_idx = group.sample(n=n_test, random_state=random_state).index
                test_indices.extend(group_test_idx)
            
            test_df = customer_df.loc[test_indices]
            control_df = customer_df.drop(test_indices)
        else:
            # Simple random split
            shuffled = customer_df.sample(frac=1, random_state=random_state)
            n_test = int(len(shuffled) * test_ratio)
            test_df = shuffled.iloc[:n_test]
            control_df = shuffled.iloc[n_test:]
        
        return test_df, control_df
    
    @staticmethod
    def matched_pair_split(
        customer_df: pd.DataFrame,
        match_on: List[str],
        test_ratio: float = 0.5,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create test/control groups using matched pair design.
        
        Args:
            customer_df: DataFrame with customer data
            match_on: Columns to match on (categorical)
            test_ratio: Proportion for test group
            random_state: Random seed
            
        Returns:
            Tuple of (test_df, control_df)
        """
        np.random.seed(random_state)
        
        test_indices = []
        control_indices = []
        
        # Group by matching columns
        grouped = customer_df.groupby(match_on)
        
        for name, group in grouped:
            shuffled = group.sample(frac=1, random_state=random_state)
            n_test = int(len(shuffled) * test_ratio)
            
            test_indices.extend(shuffled.iloc[:n_test].index)
            control_indices.extend(shuffled.iloc[n_test:].index)
        
        test_df = customer_df.loc[test_indices]
        control_df = customer_df.loc[control_indices]
        
        return test_df, control_df
    
    @staticmethod
    def geographic_split(
        customer_df: pd.DataFrame,
        test_regions: List[str],
        region_column: str = "region"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split customers by geographic region.
        
        Args:
            customer_df: DataFrame with customer data
            test_regions: List of region names for test group
            region_column: Column name containing region
            
        Returns:
            Tuple of (test_df, control_df)
        """
        test_df = customer_df[customer_df[region_column].isin(test_regions)]
        control_df = customer_df[~customer_df[region_column].isin(test_regions)]
        
        return test_df, control_df
    
    @staticmethod
    def validate_split_balance(
        test_df: pd.DataFrame,
        control_df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate the balance between test and control groups.
        
        Args:
            test_df: Test group DataFrame
            control_df: Control group DataFrame
            numeric_cols: Numeric columns to check
            categorical_cols: Categorical columns to check
            
        Returns:
            Dictionary with balance metrics
        """
        results = {
            "test_size": len(test_df),
            "control_size": len(control_df),
            "test_ratio": len(test_df) / (len(test_df) + len(control_df)),
            "numeric_balance": {},
            "categorical_balance": {},
            "is_balanced": True
        }
        
        # Check numeric columns
        if numeric_cols:
            for col in numeric_cols:
                if col in test_df.columns and col in control_df.columns:
                    test_mean = test_df[col].mean()
                    control_mean = control_df[col].mean()
                    
                    # t-test for difference
                    t_stat, p_value = stats.ttest_ind(
                        test_df[col].dropna(),
                        control_df[col].dropna()
                    )
                    
                    results["numeric_balance"][col] = {
                        "test_mean": test_mean,
                        "control_mean": control_mean,
                        "diff_pct": (test_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0,
                        "p_value": p_value,
                        "is_balanced": p_value > 0.05
                    }
                    
                    if p_value <= 0.05:
                        results["is_balanced"] = False
        
        # Check categorical columns
        if categorical_cols:
            for col in categorical_cols:
                if col in test_df.columns and col in control_df.columns:
                    test_dist = test_df[col].value_counts(normalize=True)
                    control_dist = control_df[col].value_counts(normalize=True)
                    
                    # Align distributions
                    all_cats = set(test_dist.index) | set(control_dist.index)
                    test_aligned = test_dist.reindex(all_cats, fill_value=0)
                    control_aligned = control_dist.reindex(all_cats, fill_value=0)
                    
                    # Use chi-square contingency test (handles different sample sizes)
                    test_counts = test_df[col].value_counts().reindex(all_cats, fill_value=0)
                    control_counts = control_df[col].value_counts().reindex(all_cats, fill_value=0)
                    
                    # Only run chi-square if we have valid data
                    if test_counts.sum() > 0 and control_counts.sum() > 0:
                        # Create contingency table and use chi2_contingency
                        contingency = pd.DataFrame({
                            'test': test_counts,
                            'control': control_counts
                        })
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        results["categorical_balance"][col] = {
                            "test_distribution": test_dist.to_dict(),
                            "control_distribution": control_dist.to_dict(),
                            "chi2": chi2,
                            "p_value": p_value,
                            "is_balanced": p_value > 0.05
                        }
                        
                        if p_value <= 0.05:
                            results["is_balanced"] = False
        
        return results
    
    @staticmethod
    def calculate_pre_period_correlation(
        test_series: pd.Series,
        control_series: pd.Series
    ) -> float:
        """
        Calculate correlation between test and control groups during pre-period.
        
        Args:
            test_series: Time series of test group metric
            control_series: Time series of control group metric
            
        Returns:
            Pearson correlation coefficient
        """
        # Align series by index
        aligned = pd.DataFrame({
            "test": test_series,
            "control": control_series
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        correlation = aligned["test"].corr(aligned["control"])
        return correlation
    
    @staticmethod
    def suggest_split_method(
        customer_df: pd.DataFrame,
        test_objective: str
    ) -> Dict[str, Any]:
        """
        Suggest the best split method based on data and objectives.
        
        Args:
            customer_df: Customer DataFrame
            test_objective: Description of test objective
            
        Returns:
            Dictionary with recommended method and rationale
        """
        has_region = "region" in customer_df.columns or "state" in customer_df.columns
        has_segments = "customer_attribute" in customer_df.columns
        n_customers = len(customer_df)
        
        recommendations = {
            "recommended_method": "random",
            "rationale": [],
            "alternatives": []
        }
        
        # Check for geographic tests
        if "geo" in test_objective.lower() or "region" in test_objective.lower():
            if has_region:
                recommendations["recommended_method"] = "geographic"
                recommendations["rationale"].append(
                    "Geographic split recommended for regional campaign tests"
                )
        
        # Check for segment-specific tests
        elif has_segments and ("segment" in test_objective.lower() or "customer type" in test_objective.lower()):
            recommendations["recommended_method"] = "stratified"
            recommendations["rationale"].append(
                "Stratified sampling recommended to ensure balanced segments"
            )
        
        # Default to random with stratification for large samples
        elif n_customers > 10000 and has_segments:
            recommendations["recommended_method"] = "stratified"
            recommendations["rationale"].append(
                "Large sample size with customer segments - stratified sampling recommended"
            )
        
        else:
            recommendations["recommended_method"] = "random"
            recommendations["rationale"].append(
                "Simple random split suitable for this test design"
            )
        
        # Add alternatives
        if recommendations["recommended_method"] != "random":
            recommendations["alternatives"].append("random")
        if has_region and recommendations["recommended_method"] != "geographic":
            recommendations["alternatives"].append("geographic")
        if has_segments and recommendations["recommended_method"] != "stratified":
            recommendations["alternatives"].append("stratified")
        
        return recommendations


# Singleton instance
test_design_utils = TestDesignUtils()
