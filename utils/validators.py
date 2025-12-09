"""
Data validation utilities for marketing measurement workflow.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data inputs for test design and measurement."""
    
    @staticmethod
    def validate_customer_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate customer data for test design.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check for required columns
        required_cols = ["customer_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result["is_valid"] = False
            result["errors"].append(f"Missing required columns: {missing_cols}")
        
        # Check for duplicate customer IDs
        if "customer_id" in df.columns:
            n_duplicates = df["customer_id"].duplicated().sum()
            if n_duplicates > 0:
                result["warnings"].append(
                    f"Found {n_duplicates} duplicate customer IDs"
                )
        
        # Check for null values in key columns
        for col in ["customer_id", "region", "customer_attribute"]:
            if col in df.columns:
                n_nulls = df[col].isnull().sum()
                if n_nulls > 0:
                    result["warnings"].append(
                        f"Column '{col}' has {n_nulls} null values"
                    )
        
        # Add stats
        result["stats"] = {
            "total_customers": len(df),
            "columns": list(df.columns),
            "regions": df["region"].nunique() if "region" in df.columns else 0,
            "segments": df["customer_attribute"].nunique() if "customer_attribute" in df.columns else 0
        }
        
        return result
    
    @staticmethod
    def validate_sales_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate sales data for measurement.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check for required columns
        required_cols = ["customer_id", "order_date", "order_total"]
        found_cols = [col for col in required_cols if col in df.columns]
        
        # Try alternative column names
        if "order_date" not in df.columns:
            date_cols = [c for c in df.columns if "date" in c.lower()]
            if date_cols:
                found_cols.append(date_cols[0])
                result["warnings"].append(
                    f"Using '{date_cols[0]}' as date column"
                )
        
        if "order_total" not in df.columns:
            value_cols = [c for c in df.columns if any(
                term in c.lower() for term in ["total", "amount", "revenue", "value"]
            )]
            if value_cols:
                found_cols.append(value_cols[0])
                result["warnings"].append(
                    f"Using '{value_cols[0]}' as value column"
                )
        
        missing = set(required_cols) - set(found_cols)
        if missing:
            result["is_valid"] = False
            result["errors"].append(f"Missing required columns: {list(missing)}")
        
        # Validate date column
        date_col = "order_date" if "order_date" in df.columns else None
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col])
                result["stats"]["date_range"] = {
                    "min": dates.min().strftime("%Y-%m-%d"),
                    "max": dates.max().strftime("%Y-%m-%d")
                }
            except Exception as e:
                result["warnings"].append(f"Could not parse dates: {e}")
        
        # Add basic stats
        result["stats"]["total_rows"] = len(df)
        result["stats"]["columns"] = list(df.columns)
        
        if "order_total" in df.columns:
            result["stats"]["total_revenue"] = float(df["order_total"].sum())
            result["stats"]["avg_order_value"] = float(df["order_total"].mean())
        
        return result
    
    @staticmethod
    def validate_test_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate test configuration.
        
        Args:
            config: Test configuration dictionary
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required fields
        required_fields = [
            "test_name",
            "start_date",
            "end_date",
            "split_method"
        ]
        
        for field in required_fields:
            if field not in config or not config[field]:
                result["is_valid"] = False
                result["errors"].append(f"Missing required field: {field}")
        
        # Validate dates
        if "start_date" in config and "end_date" in config:
            try:
                start = pd.to_datetime(config["start_date"])
                end = pd.to_datetime(config["end_date"])
                
                if start >= end:
                    result["is_valid"] = False
                    result["errors"].append("End date must be after start date")
                
                # Check if dates are in the future
                today = pd.Timestamp.now()
                if start < today:
                    result["warnings"].append("Start date is in the past")
                    
            except Exception as e:
                result["is_valid"] = False
                result["errors"].append(f"Invalid date format: {e}")
        
        # Validate split method
        valid_methods = ["random", "stratified", "geographic", "customer", "dma"]
        if "split_method" in config:
            if config["split_method"] not in valid_methods:
                result["warnings"].append(
                    f"Unknown split method '{config['split_method']}'. "
                    f"Valid methods: {valid_methods}"
                )
        
        # Validate test ratio
        if "test_ratio" in config:
            ratio = config["test_ratio"]
            if not 0 < ratio < 1:
                result["is_valid"] = False
                result["errors"].append("Test ratio must be between 0 and 1")
        
        return result
    
    @staticmethod
    def validate_audience_files(
        test_ids: List[str],
        control_ids: List[str],
        customer_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate test and control audience files.
        
        Args:
            test_ids: List of test customer IDs
            control_ids: List of control customer IDs
            customer_df: Full customer DataFrame
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check for overlap
        overlap = set(test_ids) & set(control_ids)
        if overlap:
            result["is_valid"] = False
            result["errors"].append(
                f"Found {len(overlap)} customers in both test and control groups"
            )
        
        # Check if all IDs exist in customer file
        all_ids = set(customer_df["customer_id"].astype(str))
        test_ids_set = set(str(id) for id in test_ids)
        control_ids_set = set(str(id) for id in control_ids)
        
        missing_test = test_ids_set - all_ids
        missing_control = control_ids_set - all_ids
        
        if missing_test:
            result["warnings"].append(
                f"{len(missing_test)} test IDs not found in customer file"
            )
        if missing_control:
            result["warnings"].append(
                f"{len(missing_control)} control IDs not found in customer file"
            )
        
        # Stats
        result["stats"] = {
            "test_count": len(test_ids),
            "control_count": len(control_ids),
            "total": len(test_ids) + len(control_ids),
            "test_ratio": len(test_ids) / (len(test_ids) + len(control_ids)) if (len(test_ids) + len(control_ids)) > 0 else 0,
            "overlap_count": len(overlap)
        }
        
        return result
    
    @staticmethod
    def validate_measurement_periods(
        pre_period: Tuple[str, str],
        post_period: Tuple[str, str],
        data_date_range: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Validate measurement periods against available data.
        
        Args:
            pre_period: (start, end) for pre-intervention period
            post_period: (start, end) for post-intervention period
            data_date_range: (min_date, max_date) of available data
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            pre_start = pd.to_datetime(pre_period[0])
            pre_end = pd.to_datetime(pre_period[1])
            post_start = pd.to_datetime(post_period[0])
            post_end = pd.to_datetime(post_period[1])
            data_min = pd.to_datetime(data_date_range[0])
            data_max = pd.to_datetime(data_date_range[1])
            
            # Check period ordering
            if pre_end >= post_start:
                result["is_valid"] = False
                result["errors"].append(
                    "Pre-period must end before post-period starts"
                )
            
            # Check against data availability
            if pre_start < data_min:
                result["warnings"].append(
                    f"Pre-period starts before available data ({data_min.strftime('%Y-%m-%d')})"
                )
            
            if post_end > data_max:
                result["warnings"].append(
                    f"Post-period ends after available data ({data_max.strftime('%Y-%m-%d')})"
                )
            
            # Check for sufficient pre-period
            pre_days = (pre_end - pre_start).days
            if pre_days < 30:
                result["warnings"].append(
                    f"Pre-period is only {pre_days} days. Recommend at least 30 days."
                )
            
            # Check for sufficient post-period
            post_days = (post_end - post_start).days
            if post_days < 14:
                result["warnings"].append(
                    f"Post-period is only {post_days} days. Recommend at least 14 days."
                )
                
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(f"Date parsing error: {e}")
        
        return result

    @staticmethod
    def validate_test_design(test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a test design configuration.
        
        Args:
            test_config: Dictionary with test configuration
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ['test_name', 'test_start_date', 'test_end_date']
        for field in required_fields:
            if field not in test_config or not test_config[field]:
                result["is_valid"] = False
                result["errors"].append(f"Missing required field: {field}")
        
        # Validate dates
        try:
            if 'test_start_date' in test_config and 'test_end_date' in test_config:
                start = pd.to_datetime(test_config['test_start_date'])
                end = pd.to_datetime(test_config['test_end_date'])
                
                if end <= start:
                    result["is_valid"] = False
                    result["errors"].append("Test end date must be after start date")
                
                duration = (end - start).days
                if duration < 14:
                    result["warnings"].append(f"Test duration is only {duration} days. Recommend at least 14 days.")
                    
        except Exception as e:
            result["warnings"].append(f"Could not validate dates: {e}")
        
        return result


# Singleton instance
data_validator = DataValidator()

# Alias for backward compatibility
TestValidator = DataValidator
