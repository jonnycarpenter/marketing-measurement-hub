"""
Data loading utilities for marketing measurement workflow.

Supports both local files (development) and GCS (production).
Set USE_GCS=true environment variable to enable GCS.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .gcs_loader import gcs_loader

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and caching of data files for the marketing measurement workflow.
    
    TODO: BigQuery Migration Guide
    ----------------------------
    To migrate to BigQuery:
    1. Create a new class `BigQueryDataLoader` inheriting from or replacing this one.
    2. Replace `_load_csv` with a method that executes BQ queries.
    3. Map the following files to BQ tables:
       - master_testing_doc.csv -> `analytics.master_testing_doc`
       - crm_sales_cust_data/customer_file.csv -> `crm.customer_dim`
       - crm_sales_cust_data/crm_sales_data_*.csv -> `crm.sales_fact`
       - product_inventory.csv -> `product.inventory_snapshot`
    4. Ensure `save_test_config` and `update_test_status` use BQ DML (INSERT/UPDATE).
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._gcs = gcs_loader
    
    def load_master_testing_doc(self) -> pd.DataFrame:
        """Load the master testing document containing all test configurations."""
        return self._load_csv("master_testing_doc.csv")
    
    def load_customer_file(self) -> pd.DataFrame:
        """Load customer data file."""
        return self._load_csv("crm_sales_cust_data/customer_file.csv")
    
    def load_sales_data(self) -> pd.DataFrame:
        """Load CRM sales data."""
        return self._load_csv("crm_sales_cust_data/crm_sales_data_2025_H12026.csv")
    
    def load_order_line_items(self) -> pd.DataFrame:
        """Load order line items data."""
        return self._load_csv("crm_sales_cust_data/order_line_items.csv")
    
    def load_product_inventory(self) -> pd.DataFrame:
        """Load product inventory data."""
        return self._load_csv("product_inventory.csv")
    
    def load_product_sku_info(self) -> pd.DataFrame:
        """Load product SKU information."""
        return self._load_csv("product_sku_info.csv")
    
    def load_promo_calendar(self) -> pd.DataFrame:
        """Load promotion calendar."""
        return self._load_csv("promo_calendar.csv")
    
    def load_measurement_history(self) -> pd.DataFrame:
        """Load measurement history."""
        return self._load_csv("test_version_history/measurement_history.csv")
    
    def load_test_audience(self, test_id: str) -> Optional[pd.DataFrame]:
        """Load test audience file for a specific test."""
        # Try new location first, fall back to old location for backwards compatibility
        filename = f"test_audience_files/{test_id}_test_audience.csv"
        try:
            return self._load_csv(filename)
        except FileNotFoundError:
            # Try old location for backwards compatibility
            old_filename = f"test_validation_files/{test_id}_test_audience.csv"
            try:
                return self._load_csv(old_filename)
            except FileNotFoundError:
                logger.warning(f"Test audience file not found: {filename}")
                return None
    
    def load_control_audience(self, test_id: str) -> Optional[pd.DataFrame]:
        """Load control audience file for a specific test."""
        # Try new location first, fall back to old location for backwards compatibility
        filename = f"test_audience_files/{test_id}_control_audience.csv"
        try:
            return self._load_csv(filename)
        except FileNotFoundError:
            # Try old location for backwards compatibility
            old_filename = f"test_validation_files/{test_id}_control_audience.csv"
            try:
                return self._load_csv(old_filename)
            except FileNotFoundError:
                logger.warning(f"Control audience file not found: {filename}")
                return None
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV file with caching. Uses GCS in production, local files in dev."""
        if filename in self._cache:
            return self._cache[filename]
        
        # Build the full relative path for GCS loader
        full_path = f"{self.data_dir}/{filename}"
        
        # Try GCS loader first (handles fallback to local automatically)
        try:
            df = self._gcs.read_csv(full_path)
            self._cache[filename] = df
            return df
        except Exception as e:
            raise FileNotFoundError(f"Data file not found: {full_path}. Error: {e}")
    
    def save_test_config(self, test_config: Dict[str, Any]) -> str:
        """Save a test configuration to the master testing doc."""
        master_df = self.load_master_testing_doc()
        
        # Convert config to DataFrame row
        new_row = pd.DataFrame([test_config])
        
        # Append to master
        updated_df = pd.concat([master_df, new_row], ignore_index=True)
        
        # Save back to file (GCS or local)
        full_path = f"{self.data_dir}/master_testing_doc.csv"
        self._gcs.write_csv(updated_df, full_path, index=False)
        
        # Clear cache
        self._cache.pop("master_testing_doc.csv", None)
        
        return test_config.get("test_id", "unknown")
    
    def update_test_status(self, test_id: str, updates: Dict[str, Any]) -> bool:
        """Update a test configuration in the master testing doc."""
        master_df = self.load_master_testing_doc()
        
        # Find the test
        mask = master_df["test_id"] == test_id
        if not mask.any():
            logger.error(f"Test not found: {test_id}")
            return False
        
        # Update fields
        for key, value in updates.items():
            if key in master_df.columns:
                master_df.loc[mask, key] = value
        
        # Save back (GCS or local)
        full_path = f"{self.data_dir}/master_testing_doc.csv"
        self._gcs.write_csv(master_df, full_path, index=False)
        
        # Clear cache
        self._cache.pop("master_testing_doc.csv", None)
        
        return True
    
    def save_audience_file(self, test_id: str, customer_ids: list, audience_type: str) -> str:
        """Save audience file (test or control) for a given test."""
        filename = f"{test_id}_{audience_type}_audience.csv"
        full_path = f"{self.data_dir}/test_audience_files/{filename}"
        
        # Create DataFrame and save (GCS or local)
        df = pd.DataFrame({"customer_id": customer_ids})
        self._gcs.write_csv(df, full_path, index=False)
        
        return full_path
    
    def get_test_summary(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a specific test."""
        master_df = self.load_master_testing_doc()
        test_row = master_df[master_df["test_id"] == test_id]
        
        if test_row.empty:
            return None
        
        return test_row.iloc[0].to_dict()
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
    
    def log_measurement_result(self, test_id: str, measurement_results: Dict[str, Any]) -> str:
        """
        Log a measurement result to the measurement history file.
        This creates a version history of all measurement runs.
        
        Args:
            test_id: The test ID
            measurement_results: Dict containing measurement results
            
        Returns:
            The history_id of the logged entry
        """
        from datetime import datetime
        
        # Load existing history
        try:
            history_df = self.load_measurement_history()
        except FileNotFoundError:
            # Create new history file with columns
            history_df = pd.DataFrame(columns=[
                "history_id", "test_id", "version", "measurement_run_date",
                "measurement_method", "lift_absolute", "lift_percent", "p_value",
                "confidence_level", "ci_lower", "ci_upper", "significant",
                "measurement_data_file", "notes"
            ])
        
        # Determine version number for this test
        test_history = history_df[history_df["test_id"] == test_id]
        version = len(test_history) + 1
        
        # Generate history ID
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        history_id = f"HIST-{timestamp}"
        
        # Create new history entry
        new_entry = {
            "history_id": history_id,
            "test_id": test_id,
            "version": version,
            "measurement_run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "measurement_method": measurement_results.get("measurement_method", ""),
            "lift_absolute": measurement_results.get("lift_absolute"),
            "lift_percent": measurement_results.get("lift_percent"),
            "p_value": measurement_results.get("p_value"),
            "confidence_level": measurement_results.get("confidence_level"),
            "ci_lower": measurement_results.get("ci_lower"),
            "ci_upper": measurement_results.get("ci_upper"),
            "significant": "Yes" if measurement_results.get("significant") else "No",
            "measurement_data_file": measurement_results.get("measurement_data_file", ""),
            "notes": measurement_results.get("notes", "")
        }
        
        # Append to history
        new_row = pd.DataFrame([new_entry])
        updated_df = pd.concat([history_df, new_row], ignore_index=True)
        
        # Save to file (GCS or local)
        full_path = f"{self.data_dir}/test_version_history/measurement_history.csv"
        self._gcs.write_csv(updated_df, full_path, index=False)
        
        # Clear cache
        self._cache.pop("test_version_history/measurement_history.csv", None)
        
        logger.info(f"Logged measurement result for test {test_id}, version {version}, history_id {history_id}")
        
        return history_id
    
    def get_measurement_history(self, test_id: str) -> pd.DataFrame:
        """
        Get all measurement history entries for a specific test.
        
        Args:
            test_id: The test ID
            
        Returns:
            DataFrame with all measurement runs for this test
        """
        try:
            history_df = self.load_measurement_history()
            return history_df[history_df["test_id"] == test_id].sort_values("version", ascending=False)
        except FileNotFoundError:
            return pd.DataFrame()


# Singleton instance for easy access
data_loader = DataLoader()
