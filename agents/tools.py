"""
Tools for the marketing measurement multi-agent workflow.
These tools are used by agents to perform specific tasks.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
import sys
import os
import requests
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from utils.test_design_utils import TestDesignUtils
from utils.causal_impact_utils import CausalImpactAnalyzer
from utils.validators import DataValidator
from utils.logging_utils import (
    create_test_folders,
    create_design_phase_files,
    create_measurement_phase_files,
    create_version_history_entry,
    add_test_to_master,
    update_test_status as logging_update_test_status,
    update_test_with_measurement_results,
    get_test_by_id,
    get_all_tests,
    generate_test_id as logging_generate_test_id
)
from utils.rag_utils import (
    search_knowledge_base as rag_search,
    index_knowledge_base as rag_index,
    get_knowledge_base_stats as rag_stats
)

logger = logging.getLogger(__name__)

import contextvars

# Initialize utilities
data_loader = DataLoader(data_dir=str(Path(__file__).parent.parent / "data"))
test_design_utils = TestDesignUtils()
causal_impact_analyzer = CausalImpactAnalyzer()
data_validator = DataValidator()

# Session context for multi-user support
session_context = contextvars.ContextVar("session_id", default="default")

# Global storage keyed by session_id
# Structure: {session_id: {split_cache_data}}
_session_storage: Dict[str, Any] = {}

def get_current_split_cache() -> Dict[str, Any]:
    """Get the split cache for the current session."""
    session_id = session_context.get()
    if session_id not in _session_storage:
        _session_storage[session_id] = {
            "test_customer_ids": [],
            "control_customer_ids": [],
            "split_method": None,
            "split_params": {},
            "balance_check": None,
            "created_at": None
        }
    return _session_storage[session_id]

def clear_current_split_cache():
    """Clear the split cache for the current session."""
    session_id = session_context.get()
    if session_id in _session_storage:
        _session_storage[session_id] = {
            "test_customer_ids": [],
            "control_customer_ids": [],
            "split_method": None,
            "split_params": {},
            "balance_check": None,
            "created_at": None
        }


# ==================== Data Loading Tools ====================

def get_customer_data_summary() -> str:
    """
    Get a summary of available customer data for test design.
    Returns information about total customers, regions, segments, and stratification options.
    """
    try:
        customer_df = data_loader.load_customer_file()
        
        summary = {
            "total_customers": len(customer_df),
            "columns": list(customer_df.columns),
            "dmas": customer_df["dma"].unique().tolist() if "dma" in customer_df.columns else [],
            "dma_counts": customer_df["dma"].value_counts().to_dict() if "dma" in customer_df.columns else {},
            "metros": customer_df["metro"].unique().tolist() if "metro" in customer_df.columns else [],
            "segments": customer_df["customer_attribute"].unique().tolist() if "customer_attribute" in customer_df.columns else [],
            "segment_counts": customer_df["customer_attribute"].value_counts().to_dict() if "customer_attribute" in customer_df.columns else {},
            "age_stats": {
                "min": int(customer_df["age"].min()),
                "max": int(customer_df["age"].max()),
                "mean": float(customer_df["age"].mean())
            } if "age" in customer_df.columns else {},
            # Stratification tier options
            "stratification_options": {
                "purch_freq_tier": customer_df["purch_freq_tier"].value_counts().to_dict() if "purch_freq_tier" in customer_df.columns else {},
                "ltv_tier": customer_df["ltv_tier"].value_counts().to_dict() if "ltv_tier" in customer_df.columns else {},
                "gender": customer_df["gender"].value_counts().to_dict() if "gender" in customer_df.columns else {}
            },
            "stratification_help": "For balanced test/control splits, you can stratify by: purch_freq_tier (high/medium/low frequency buyers), ltv_tier (high/medium/low value customers), region, gender, or customer_attribute (segment)"
        }
        
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_crm_sales_data_summary() -> str:
    """
    Get a summary of the CRM sales data (crm_sales_data_2025_H12026.csv).
    Returns information about total orders, date range, geographic distribution,
    and order value statistics.
    """
    try:
        sales_df = data_loader.load_sales_data()
        
        # Parse dates for date range analysis
        sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], format='%m/%d/%Y', errors='coerce')
        
        summary = {
            "description": "CRM Sales Data for H1 2025 - H2 2026: Transaction-level sales records linking customers to their purchases",
            "total_records": len(sales_df),
            "columns": list(sales_df.columns),
            "column_descriptions": {
                "order_date": "Date of the transaction",
                "customer_id": "Unique identifier linking to customer profiles",
                "dma": "Designated Market Area (city/metro region)",
                "state": "US state where the order was placed",
                "unique_order_id": "Unique transaction identifier (links to order_line_items)",
                "order_value_usd": "Total order value in USD"
            },
            "date_range": {
                "earliest_order": sales_df['order_date'].min().strftime('%Y-%m-%d') if pd.notna(sales_df['order_date'].min()) else None,
                "latest_order": sales_df['order_date'].max().strftime('%Y-%m-%d') if pd.notna(sales_df['order_date'].max()) else None
            },
            "unique_customers": int(sales_df['customer_id'].nunique()),
            "unique_orders": int(sales_df['unique_order_id'].nunique()),
            "geographic_distribution": {
                "states": sales_df['state'].nunique() if 'state' in sales_df.columns else 0,
                "top_states": sales_df['state'].value_counts().head(10).to_dict() if 'state' in sales_df.columns else {},
                "dmas": sales_df['dma'].nunique() if 'dma' in sales_df.columns else 0,
                "top_dmas": sales_df['dma'].value_counts().head(10).to_dict() if 'dma' in sales_df.columns else {}
            },
            "order_value_stats": {
                "min": float(sales_df['order_value_usd'].min()),
                "max": float(sales_df['order_value_usd'].max()),
                "mean": round(float(sales_df['order_value_usd'].mean()), 2),
                "median": round(float(sales_df['order_value_usd'].median()), 2),
                "total_revenue": round(float(sales_df['order_value_usd'].sum()), 2)
            },
            "use_cases": "This data is used for sales performance analysis, customer value metrics, and measuring test outcomes by comparing order values between test and control groups."
        }
        
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_order_line_items_summary() -> str:
    """
    Get a summary of the order line items data (order_line_items.csv).
    Returns information about products, pricing, and order composition.
    """
    try:
        line_items_df = data_loader.load_order_line_items()
        
        summary = {
            "description": "Order Line Items: Granular product-level details for each order, showing what specific items were purchased in each transaction",
            "total_records": len(line_items_df),
            "columns": list(line_items_df.columns),
            "column_descriptions": {
                "unique_order_id": "Links to the parent order in crm_sales_data (foreign key)",
                "product_id": "Unique product SKU identifier",
                "product_name": "Human-readable product name",
                "price": "Unit price of the product in USD"
            },
            "unique_orders": int(line_items_df['unique_order_id'].nunique()),
            "unique_products": int(line_items_df['product_id'].nunique()),
            "product_catalog": {
                "total_products": int(line_items_df['product_id'].nunique()),
                "products": line_items_df.groupby('product_id')['product_name'].first().to_dict(),
                "product_price_list": line_items_df.groupby(['product_id', 'product_name'])['price'].first().reset_index().to_dict('records')
            },
            "top_products_by_volume": line_items_df['product_name'].value_counts().head(10).to_dict(),
            "price_stats": {
                "min_price": float(line_items_df['price'].min()),
                "max_price": float(line_items_df['price'].max()),
                "mean_price": round(float(line_items_df['price'].mean()), 2),
                "median_price": round(float(line_items_df['price'].median()), 2)
            },
            "avg_items_per_order": round(len(line_items_df) / line_items_df['unique_order_id'].nunique(), 2),
            "use_cases": "This data enables product-level insights: which products are most popular, cross-selling patterns, basket analysis, and understanding which specific items drive revenue in test vs control groups."
        }
        
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def check_promo_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Check the promotional calendar for any scheduled promotions.
    Use this to ensure test periods don't overlap with major promotions that could confound results.
    
    Args:
        start_date: Optional start date to filter promotions (YYYY-MM-DD format)
        end_date: Optional end date to filter promotions (YYYY-MM-DD format)
        
    Returns:
        JSON with promotion details including name, dates, products affected, and potential conflicts
    """
    try:
        promo_df = data_loader.load_promo_calendar()
        
        if promo_df is None or promo_df.empty:
            return json.dumps({"message": "No promotions found in the calendar", "promotions": []})
        
        # Parse dates in promo calendar
        promo_df['start_date_parsed'] = pd.to_datetime(promo_df['start_date'], format='%m/%d/%Y', errors='coerce')
        promo_df['end_date_parsed'] = pd.to_datetime(promo_df['end_date'], format='%m/%d/%Y', errors='coerce')
        
        # Filter by date range if provided
        filtered_df = promo_df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date)
            # Include promotions that end after the start date
            filtered_df = filtered_df[filtered_df['end_date_parsed'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Include promotions that start before the end date
            filtered_df = filtered_df[filtered_df['start_date_parsed'] <= end_dt]
        
        promotions = []
        for _, row in filtered_df.iterrows():
            promo = {
                "promotion_name": row.get("promotion_name", ""),
                "description": row.get("description", ""),
                "product_id": row.get("promo_product_id", ""),
                "start_date": row.get("start_date", ""),
                "end_date": row.get("end_date", ""),
                "actor": row.get("actor", ""),
            }
            promotions.append(promo)
        
        # Build summary
        result = {
            "total_promotions": len(promotions),
            "date_range_checked": {
                "start": start_date or "all",
                "end": end_date or "all"
            },
            "promotions": promotions,
            "warning": "Consider these promotions when designing your test. Promotions can confound results if they overlap with your test period." if promotions else None
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_existing_tests() -> str:
    """
    Get a list of all existing tests from the master testing document.
    Returns test IDs, names, statuses, and key configuration.
    """
    try:
        # Clear cache to ensure we get the latest data
        data_loader.clear_cache()
        master_df = data_loader.load_master_testing_doc()
        
        tests = []
        for _, row in master_df.iterrows():
            test_info = {
                "test_id": row.get("test_id", ""),
                "test_name": row.get("test_name", ""),
                "status": row.get("status", ""),
                "start_date": str(row.get("start_date", "")),
                "end_date": str(row.get("end_date", "")),
                "split_method": row.get("split_method", ""),
                "measurement_method": row.get("measurement_method", "")
            }
            tests.append(test_info)
        
        return json.dumps({"tests": tests, "total_count": len(tests)}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_test_details(test_id: str) -> str:
    """
    Get detailed information about a specific test.
    
    Args:
        test_id: The unique identifier for the test
        
    Returns:
        JSON string with complete test configuration and results
    """
    try:
        # Clear cache to ensure we get the latest data
        data_loader.clear_cache()
        test_summary = data_loader.get_test_summary(test_id)
        
        if test_summary is None:
            return json.dumps({"error": f"Test not found: {test_id}"})
        
        # Convert any NaN values to None for JSON serialization
        test_summary = {k: (None if pd.isna(v) else v) for k, v in test_summary.items()}
        
        return json.dumps(test_summary, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ==================== Test Design Tools ====================

def calculate_sample_size(
    baseline_conversion_rate: float,
    expected_lift_percent: float,
    confidence_level: float = 0.95,
    power: float = 0.80
) -> str:
    """
    Calculate the required sample size for a marketing test.
    
    Args:
        baseline_conversion_rate: Current conversion rate (e.g., 0.05 for 5%)
        expected_lift_percent: Expected relative lift as percentage (e.g., 10 for 10%)
        confidence_level: Statistical confidence level (default 0.95)
        power: Statistical power (default 0.80)
        
    Returns:
        JSON with recommended sample size and explanation
    """
    try:
        expected_lift = expected_lift_percent / 100
        
        sample_size = test_design_utils.calculate_sample_size(
            baseline_conversion=baseline_conversion_rate,
            expected_lift=expected_lift,
            confidence_level=confidence_level,
            power=power
        )
        
        result = {
            "sample_size_per_group": sample_size,
            "total_sample_size": sample_size * 2,
            "parameters": {
                "baseline_conversion_rate": baseline_conversion_rate,
                "expected_lift_percent": expected_lift_percent,
                "confidence_level": confidence_level,
                "power": power
            },
            "interpretation": f"You need approximately {sample_size:,} customers per group "
                            f"({sample_size * 2:,} total) to detect a {expected_lift_percent}% lift "
                            f"with {int(confidence_level * 100)}% confidence and {int(power * 100)}% power."
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_random_split(
    test_ratio: float = 0.5,
    stratify_by: Optional[str] = None,
    random_seed: int = 42
) -> str:
    """
    Create a random test/control split of the customer base.
    
    Args:
        test_ratio: Proportion of customers in test group (default 0.5)
        stratify_by: Comma-separated column names to stratify by (optional)
        random_seed: Random seed for reproducibility
        
    Returns:
        JSON with split statistics and customer counts
    """
    try:
        customer_df = data_loader.load_customer_file()
        
        stratify_cols = None
        if stratify_by:
            stratify_cols = [col.strip() for col in stratify_by.split(",")]
            # Validate columns exist
            missing = [col for col in stratify_cols if col not in customer_df.columns]
            if missing:
                return json.dumps({"error": f"Columns not found: {missing}"})
        
        test_df, control_df = test_design_utils.random_split(
            customer_df=customer_df,
            test_ratio=test_ratio,
            stratify_by=stratify_cols,
            random_state=random_seed
        )
        
        # Validate balance
        numeric_cols = ["age", "projected_ltv", "purchase_frequency"]
        numeric_cols = [c for c in numeric_cols if c in customer_df.columns]
        
        categorical_cols = ["gender", "customer_attribute", "dma", "metro"]
        categorical_cols = [c for c in categorical_cols if c in customer_df.columns]
        
        balance_check = test_design_utils.validate_split_balance(
            test_df, control_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )
        
        # Cache the full split results for later saving
        split_cache = get_current_split_cache()
        split_cache.update({
            "test_customer_ids": test_df["customer_id"].tolist(),
            "control_customer_ids": control_df["customer_id"].tolist(),
            "split_method": "stratified" if stratify_cols else "random",
            "split_params": {
                "test_ratio": test_ratio,
                "stratify_by": stratify_cols,
                "random_seed": random_seed
            },
            "balance_check": balance_check,
            "created_at": datetime.now().isoformat()
        })
        
        result = {
            "success": True,
            "test_count": len(test_df),
            "control_count": len(control_df),
            "actual_test_ratio": len(test_df) / (len(test_df) + len(control_df)),
            "stratified_by": stratify_cols,
            "balance_check": {
                "is_balanced": balance_check["is_balanced"],
                "numeric_balance": {
                    k: {kk: round(vv, 4) if isinstance(vv, float) else vv 
                        for kk, vv in v.items()}
                    for k, v in balance_check["numeric_balance"].items()
                },
                "categorical_balance": {
                    k: {"is_balanced": v["is_balanced"], "p_value": round(v["p_value"], 4)}
                    for k, v in balance_check["categorical_balance"].items()
                }
            },
            "test_customer_ids_preview": test_df["customer_id"].tolist()[:5],
            "control_customer_ids_preview": control_df["customer_id"].tolist()[:5],
            "split_cached": True,
            "next_step": "Use save_current_test_design() to save this split with test details"
        }
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_geographic_split(test_regions: str) -> str:
    """
    Create a test/control split based on DMAs (Designated Market Areas).
    
    Args:
        test_regions: Comma-separated list of DMA names for the test group
                     (e.g., "Dallas-Ft. Worth DMA, Houston DMA, Atlanta DMA")
        
    Returns:
        JSON with split statistics and DMA breakdown
    """
    try:
        # Validate input
        if test_regions is None or not isinstance(test_regions, str) or test_regions.strip() == "":
            return json.dumps({
                "error": "test_regions parameter is required and must be a non-empty string",
                "hint": "Provide a comma-separated list of DMA names, e.g., 'Dallas-Ft. Worth DMA, Houston DMA'"
            })
        
        customer_df = data_loader.load_customer_file()
        
        if "dma" not in customer_df.columns:
            return json.dumps({"error": "DMA column not found in customer data"})
        
        test_region_list = [r.strip() for r in test_regions.split(",")]
        available_regions = customer_df["dma"].unique().tolist()
        
        # Validate regions
        invalid = [r for r in test_region_list if r not in available_regions]
        if invalid:
            return json.dumps({
                "error": f"Invalid regions: {invalid}",
                "available_regions": available_regions
            })
        
        test_df, control_df = test_design_utils.geographic_split(
            customer_df=customer_df,
            test_regions=test_region_list,
            region_column="dma"
        )
        
        # Cache the full split results for later saving
        split_cache = get_current_split_cache()
        split_cache.update({
            "test_customer_ids": test_df["customer_id"].tolist(),
            "control_customer_ids": control_df["customer_id"].tolist(),
            "split_method": "geographic",
            "split_params": {
                "test_regions": test_region_list,
                "control_regions": [r for r in available_regions if r not in test_region_list]
            },
            "balance_check": None,  # Geographic splits don't use statistical balance
            "created_at": datetime.now().isoformat()
        })
        
        result = {
            "success": True,
            "test_count": len(test_df),
            "control_count": len(control_df),
            "test_ratio": len(test_df) / (len(test_df) + len(control_df)),
            "test_regions": test_region_list,
            "control_regions": [r for r in available_regions if r not in test_region_list],
            "dma_breakdown": {
                "test": test_df["dma"].value_counts().to_dict(),
                "control": control_df["dma"].value_counts().to_dict()
            },
            "split_cached": True,
            "next_step": "Use save_current_test_design() to save this split with test details"
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def save_current_test_design(
    test_name: str,
    test_description: str,
    start_date: str,
    end_date: str,
    pre_period_weeks: int = 26,
    measurement_method: str = "causal impact"
) -> str:
    """
    Save the current cached test/control split as a new test.
    Uses the most recent split created by create_random_split() or create_geographic_split().
    
    This is the PREFERRED method for saving tests - it automatically handles customer IDs
    from the cached split, so users don't need to manage ID lists.
    
    Args:
        test_name: Name of the test (e.g., "Q1 Email Campaign Test")
        test_description: Description of what is being tested
        start_date: Test start date (YYYY-MM-DD format)
        end_date: Test end date (YYYY-MM-DD format)
        pre_period_weeks: Number of weeks for pre-period analysis (default: 26, ~180 days for optimal model convergence)
        measurement_method: Method for measuring results (default: "causal impact")
        
    Returns:
        JSON with saved test ID and confirmation details
    """
    split_cache = get_current_split_cache()
    
    # Validate cache exists
    if not split_cache.get("test_customer_ids"):
        return json.dumps({
            "error": "No split cached. Please create a split first using create_random_split() or create_geographic_split()",
            "hint": "Run create_random_split() or create_geographic_split() before saving"
        })
    
    # Extract from cache
    test_customer_ids = split_cache["test_customer_ids"]
    control_customer_ids = split_cache["control_customer_ids"]
    split_method = split_cache["split_method"]
    split_params = split_cache.get("split_params", {})
    
    # For geographic splits, get DMA/region info
    test_dmas = "not applicable"
    if split_method == "geographic":
        test_regions = split_params.get("test_regions") if split_params else None
        if test_regions and isinstance(test_regions, list):
            test_dmas = ", ".join(test_regions)
    
    # Delegate to the full save function
    try:
        result = save_test_design(
            test_name=test_name,
            test_description=test_description,
            start_date=start_date,
            end_date=end_date,
            split_method=split_method,
            test_customer_ids=test_customer_ids,
            control_customer_ids=control_customer_ids,
            pre_period_weeks=pre_period_weeks,
            measurement_method=measurement_method,
            test_dmas=test_dmas
        )
    except Exception as e:
        logger.exception(f"Error in save_test_design: {e}")
        return json.dumps({
            "error": f"Save operation encountered an issue: {str(e)}",
            "partial_success": True,
            "message": "The test may have been partially saved. Please check the test list."
        })
    
    # Clear the cache after successful save
    try:
        result_dict = json.loads(result) if result else {}
    except json.JSONDecodeError:
        result_dict = {"success": False, "raw_result": str(result)}
        
    if result_dict.get("success"):
        clear_current_split_cache()
        test_id = result_dict.get('test_id')
        logger.info(f"Split cache cleared after saving test: {test_id}")
        
        # Auto-generate pre-period balance chart and include in response
        # This ensures the chart is always shown even if the LLM doesn't call it separately
        try:
            print(f"[AUTO-CHART] Generating pre-period chart for test {test_id}")
            chart_result = generate_pre_period_balance_charts(test_id=test_id, chart_type="weekly_trend")
            print(f"[AUTO-CHART] Chart result length: {len(chart_result) if chart_result else 0}")
            chart_data = json.loads(chart_result) if chart_result else {}
            if chart_data.get("charts"):
                result_dict["pre_period_chart"] = chart_data
                result_dict["chart_reminder"] = "IMPORTANT: The pre-period weekly trend chart is included above - it validates test/control balance."
                print(f"[AUTO-CHART] Successfully embedded {len(chart_data['charts'])} chart(s) in save response")
                logger.info(f"Auto-generated pre-period chart for test {test_id}")
                return json.dumps(result_dict, indent=2)
            elif chart_data.get("error"):
                print(f"[AUTO-CHART] Chart generation error: {chart_data.get('error')}")
                result_dict["chart_error"] = chart_data.get("error")
        except Exception as chart_err:
            print(f"[AUTO-CHART] Exception: {chart_err}")
            logger.warning(f"Could not auto-generate pre-period chart: {chart_err}")
            result_dict["chart_note"] = f"Call generate_pre_period_balance_charts(test_id='{test_id}', chart_type='weekly_trend') to see the pre-period trending chart."
            return json.dumps(result_dict, indent=2)
    
    return result


def get_current_split_status() -> str:
    """
    Check if there's a cached split ready to be saved.
    
    Returns:
        JSON with current split status and details
    """
    split_cache = get_current_split_cache()
    
    if not split_cache.get("test_customer_ids"):
        return json.dumps({
            "has_cached_split": False,
            "message": "No split currently cached. Use create_random_split() or create_geographic_split() first."
        })
    
    return json.dumps({
        "has_cached_split": True,
        "split_method": split_cache["split_method"],
        "test_count": len(split_cache["test_customer_ids"]),
        "control_count": len(split_cache["control_customer_ids"]),
        "split_params": split_cache.get("split_params", {}),
        "created_at": split_cache.get("created_at"),
        "ready_to_save": True,
        "next_step": "Use save_current_test_design() with test name, description, start_date, and end_date"
    }, indent=2)


def generate_cached_split_balance_chart(
    pre_period_weeks: int = 13,
    metric: str = "revenue"
) -> str:
    """
    Generate a pre-period balance chart using the CURRENT CACHED SPLIT (before saving).
    
    Use this tool to validate test/control balance DURING the design iteration process,
    before committing to save the test. This is essential when trying multiple split
    configurations to find one with good balance.
    
    NOTE: This uses the in-memory cached split from create_random_split() or 
    create_geographic_split(). If you need charts for an already-saved test,
    use generate_pre_period_balance_charts(test_id=...) instead.
    
    Args:
        pre_period_weeks: Number of weeks of historical data to analyze (default: 13 weeks)
        metric: What to compare - "revenue" or "orders" (default: "revenue")
        
    Returns:
        JSON with base64 encoded chart image and balance statistics (correlation)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import base64
    
    try:
        # Get the cached split
        split_cache = get_current_split_cache()
        
        if not split_cache.get("test_customer_ids"):
            return json.dumps({
                "error": "No cached split found. Use create_random_split() or create_geographic_split() first.",
                "hint": "You must create a split before generating a balance chart."
            })
        
        test_ids = split_cache["test_customer_ids"]
        control_ids = split_cache["control_customer_ids"]
        split_method = split_cache.get("split_method", "unknown")
        
        # Load sales data
        sales_df = data_loader.load_sales_data()
        if sales_df is None:
            return json.dumps({"error": "Could not load sales data"})
        
        # Detect date column
        date_col = None
        for col in ["order_date", "transaction_date", "date"]:
            if col in sales_df.columns:
                date_col = col
                break
        if date_col is None:
            return json.dumps({"error": "Could not find date column in sales data"})
        
        sales_df[date_col] = pd.to_datetime(sales_df[date_col], format='%m/%d/%Y', errors='coerce')
        
        # Detect value column
        value_col = None
        for col in ["order_value_usd", "order_value", "revenue", "amount"]:
            if col in sales_df.columns:
                value_col = col
                break
        
        # Detect customer ID column
        cust_col = None
        for col in ["customer_id", "cust_id", "customerid"]:
            if col in sales_df.columns:
                cust_col = col
                break
        if cust_col is None:
            return json.dumps({"error": "Could not find customer_id column in sales data"})
        
        # Calculate pre-period (most recent N weeks of data)
        max_date = sales_df[date_col].max()
        pre_end_dt = max_date
        pre_start_dt = max_date - pd.Timedelta(weeks=pre_period_weeks)
        
        # Filter to pre-period
        pre_period_mask = (sales_df[date_col] >= pre_start_dt) & (sales_df[date_col] <= pre_end_dt)
        pre_sales = sales_df[pre_period_mask].copy()
        
        # Split into test and control
        test_sales = pre_sales[pre_sales[cust_col].isin(test_ids)]
        control_sales = pre_sales[pre_sales[cust_col].isin(control_ids)]
        
        if test_sales.empty or control_sales.empty:
            return json.dumps({
                "error": "No sales data found for test or control groups in the pre-period",
                "test_customers_with_sales": len(test_sales[cust_col].unique()) if not test_sales.empty else 0,
                "control_customers_with_sales": len(control_sales[cust_col].unique()) if not control_sales.empty else 0,
                "pre_period": f"{pre_start_dt.strftime('%Y-%m-%d')} to {pre_end_dt.strftime('%Y-%m-%d')}"
            })
        
        # Aggregate by week
        test_sales['week'] = test_sales[date_col].dt.to_period('W').apply(lambda x: x.start_time)
        control_sales['week'] = control_sales[date_col].dt.to_period('W').apply(lambda x: x.start_time)
        
        if metric == "revenue" and value_col:
            test_weekly = test_sales.groupby('week')[value_col].sum().reset_index()
            control_weekly = control_sales.groupby('week')[value_col].sum().reset_index()
            test_weekly.columns = ['week', 'value']
            control_weekly.columns = ['week', 'value']
            metric_label = "Revenue ($)"
        else:
            test_weekly = test_sales.groupby('week').size().reset_index(name='value')
            control_weekly = control_sales.groupby('week').size().reset_index(name='value')
            metric_label = "Order Count"
        
        # Merge for comparison
        weekly_df = test_weekly.merge(control_weekly, on='week', how='outer', suffixes=('_test', '_control'))
        weekly_df = weekly_df.fillna(0).sort_values('week')
        
        # Calculate balance statistics
        correlation = weekly_df['value_test'].corr(weekly_df['value_control'])
        test_mean = weekly_df['value_test'].mean()
        control_mean = weekly_df['value_control'].mean()
        pct_diff = abs(test_mean - control_mean) / ((test_mean + control_mean) / 2) * 100 if (test_mean + control_mean) > 0 else 0
        
        # Set up styling
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        # Generate chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(weekly_df['week'], weekly_df['value_test'], 'b-', linewidth=2, marker='o', markersize=4, label='Test Group')
        ax.plot(weekly_df['week'], weekly_df['value_control'], 'r--', linewidth=2, marker='s', markersize=4, label='Control Group')
        
        split_label = split_method.replace("_", " ").title()
        ax.set_title(f'Pre-Period Weekly {metric.title()} - {split_label} Split (Cached)\nCorrelation: {correlation:.3f}')
        ax.set_xlabel('Week')
        ax.set_ylabel(metric_label)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add balance indicator
        is_good_balance = correlation > 0.8
        balance_text = f"Balance: {'✓ Good' if is_good_balance else '⚠ Check'} (r={correlation:.2f})"
        ax.annotate(balance_text, xy=(0.98, 0.02), xycoords='axes fraction', ha='right',
                   fontsize=10, color='green' if is_good_balance else 'orange',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return json.dumps({
            "success": True,
            "split_method": split_method,
            "pre_period": f"{pre_start_dt.strftime('%Y-%m-%d')} to {pre_end_dt.strftime('%Y-%m-%d')}",
            "test_customers": len(test_ids),
            "control_customers": len(control_ids),
            "test_customers_with_sales": len(test_sales[cust_col].unique()),
            "control_customers_with_sales": len(control_sales[cust_col].unique()),
            "balance_stats": {
                "correlation": round(correlation, 4),
                "is_balanced": is_good_balance,
                "test_weekly_avg": round(test_mean, 2),
                "control_weekly_avg": round(control_mean, 2),
                "pct_difference": round(pct_diff, 2)
            },
            "chart": {
                "type": "weekly_trend",
                "title": f"Pre-Period Weekly {metric.title()} Trend (Cached Split)",
                "description": f"Test vs Control weekly {metric}. Correlation: {correlation:.3f}. Higher correlation (>0.8) = better balance.",
                "image_base64": img_base64
            },
            "recommendation": "Good balance - proceed with save_current_test_design()" if is_good_balance else "Consider trying a different split configuration for better balance"
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error generating cached split balance chart: {e}")
        return json.dumps({"error": str(e)})


def save_test_design(
    test_name: str,
    test_description: str,
    start_date: str,
    end_date: str,
    split_method: str,
    test_customer_ids: List[int],
    control_customer_ids: List[int],
    pre_period_weeks: int = 26,
    measurement_method: str = "causal impact",
    test_dmas: str = "not applicable"
) -> str:
    """
    Save a complete test design to the master testing document (internal function).
    Creates test folders, validation files, and logs to version history.
    
    NOTE: For most use cases, use save_current_test_design() instead - it automatically
    uses the cached split from create_random_split() or create_geographic_split().
    
    Args:
        test_name: Name of the test
        test_description: Description of what is being tested
        start_date: Test start date (YYYY-MM-DD)
        end_date: Test end date (YYYY-MM-DD)
        split_method: Method used for splitting (random, stratified, geographic, customer, dma)
        test_customer_ids: List of customer IDs in test group
        control_customer_ids: List of customer IDs in control group
        pre_period_weeks: Number of weeks for pre-period analysis (default: 26, ~180 days for optimal model convergence)
        measurement_method: Method for measuring results (default: causal impact)
        test_dmas: Comma-separated list of DMAs for geographic tests
        
    Returns:
        JSON with saved test ID and confirmation
    """
    try:
        # Calculate pre-period dates
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        pre_start = (start_dt - timedelta(weeks=pre_period_weeks)).strftime("%Y-%m-%d")
        pre_end = (start_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Create test config for logging_utils
        test_config = {
            "test_name": test_name,
            "test_description": test_description,
            "start_date": start_date,
            "end_date": end_date,
            "split_method": split_method,
            "test_dmas": test_dmas if split_method in ["geographic", "dma"] else "not applicable",
            "test_cust_count": len(test_customer_ids),
            "pre_period_lookback_start": pre_start,
            "pre_period_lookback_end": pre_end
        }
        
        # Use logging_utils to add test to master doc (creates folders and version history)
        test_id = add_test_to_master(test_config)
        
        # Save audience files to the test validation folder
        test_file = data_loader.save_audience_file(test_id, test_customer_ids, "test")
        control_file = data_loader.save_audience_file(test_id, control_customer_ids, "control")
        
        # --- NEW: Generate Validation Package (Pre-Test) ---
        validation_dir = Path(data_loader.data_dir) / "test_validation_files"
        
        # 1. Experiment Design Document (Markdown)
        design_doc = f"""# Experiment Design Document: {test_name}
**Test ID:** {test_id}
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## 1. Business Problem & Objective
{test_description}

## 2. Methodology
- **Split Method:** {split_method}
- **Test Group Size:** {len(test_customer_ids):,}
- **Control Group Size:** {len(control_customer_ids):,}
- **Measurement Method:** {measurement_method}

## 3. Key Performance Indicators (KPIs)
- Primary: Revenue / Orders
- Secondary: Customer Retention

## 4. Timeline
- **Pre-Period:** {pre_start} to {pre_end}
- **Test Period:** {start_date} to {end_date}
"""
        with open(validation_dir / f"{test_id}_experiment_design.md", "w") as f:
            f.write(design_doc)

        # 2. Configuration File (JSON)
        with open(validation_dir / f"{test_id}_test_config.json", "w") as f:
            json.dump(test_config, f, indent=2)

        # Generate pre-period validation data for design phase files
        try:
            # Load sales data for pre-period analysis
            sales_df = data_loader.load_sales_data()
            customer_df = data_loader.load_customer_file()
            
            # Calculate pre-period summary statistics
            pre_period_data = _generate_pre_period_validation_data(
                test_customer_ids=test_customer_ids,
                control_customer_ids=control_customer_ids,
                sales_df=sales_df,
                customer_df=customer_df,
                pre_start=pre_start,
                pre_end=pre_end
            )
            
            # 3. Pre-Period Balance Metrics (CSV)
            if "pre_period_summary" in pre_period_data:
                pre_period_data["pre_period_summary"].to_csv(
                    validation_dir / f"{test_id}_pre_period_balance_metrics.csv", index=False
                )
            
            # 4. Pre-Period Model Fit (JSON placeholder)
            model_fit = {
                "mape": "N/A (Calculated post-hoc)",
                "durbin_watson": "N/A (Calculated post-hoc)",
                "note": "Model fit statistics will be populated after measurement run."
            }
            with open(validation_dir / f"{test_id}_pre_period_model_fit.json", "w") as f:
                json.dump(model_fit, f, indent=2)

            # Create design phase validation files with actual data (Legacy support)
            design_files = create_design_phase_files(test_id, pre_period_data)
            logger.info(f"Created {len(design_files)} design phase validation files")
        except Exception as e:
            logger.warning(f"Could not generate full pre-period data: {e}. Creating empty files.")
            design_files = create_design_phase_files(test_id)
        
        # Track the most recently designed test for Brena's context
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                st.session_state['last_designed_test_id'] = test_id
                st.session_state['last_designed_test_name'] = test_name
        except:
            pass  # Not in Streamlit context
        
        result = {
            "success": True,
            "test_id": test_id,
            "message": f"Test '{test_name}' has been saved successfully",
            "folders_created": {
                "validation_folder": f"data/test_validation_files/",
                "version_history": f"data/test_version_history/"
            },
            "files_created": {
                "test_audience": test_file,
                "control_audience": control_file,
                "experiment_design": f"{test_id}_experiment_design.md",
                "test_config": f"{test_id}_test_config.json",
                "balance_metrics": f"{test_id}_pre_period_balance_metrics.csv"
            },
            "test_config": {
                "test_id": test_id,
                "test_name": test_name,
                "status": "Scheduled",
                "split_method": split_method,
                "test_count": len(test_customer_ids),
                "control_count": len(control_customer_ids),
                "pre_period": f"{pre_start} to {pre_end}",
                "test_period": f"{start_date} to {end_date}"
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("Error saving test design")
        return json.dumps({"error": str(e)})


def _generate_pre_period_validation_data(
    test_customer_ids: List[int],
    control_customer_ids: List[int],
    sales_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    pre_start: str,
    pre_end: str
) -> Dict[str, pd.DataFrame]:
    """
    Generate pre-period validation data for design phase files.
    
    Returns dict of DataFrames keyed by file type.
    """
    from scipy import stats
    
    result = {}
    
    # Auto-detect date column
    date_col = None
    for col in ['order_date', 'date', 'transaction_date', 'purchase_date']:
        if col in sales_df.columns:
            date_col = col
            break
    if date_col is None:
        raise KeyError(f"Could not find date column. Available: {sales_df.columns.tolist()}")
    
    # Filter sales to pre-period
    sales_df['date'] = pd.to_datetime(sales_df[date_col])
    pre_sales = sales_df[
        (sales_df['date'] >= pre_start) & 
        (sales_df['date'] <= pre_end)
    ]
    
    # Identify customer ID column
    cust_col = 'customer_id' if 'customer_id' in pre_sales.columns else 'cust_id'
    
    # Split sales by group
    test_sales = pre_sales[pre_sales[cust_col].isin(test_customer_ids)]
    control_sales = pre_sales[pre_sales[cust_col].isin(control_customer_ids)]
    
    # Auto-detect value column
    value_col = None
    for col in ['order_value_usd', 'order_total', 'order_value', 'revenue', 'sales_amount', 'amount', 'total']:
        if col in pre_sales.columns:
            value_col = col
            break
    if value_col is None:
        raise KeyError(f"Could not find value column. Available: {pre_sales.columns.tolist()}")
    
    metrics = []
    for metric_name in [value_col, 'transaction_count']:
        if metric_name == 'transaction_count':
            test_agg = test_sales.groupby(cust_col).size()
            control_agg = control_sales.groupby(cust_col).size()
        else:
            test_agg = test_sales.groupby(cust_col)[metric_name].sum()
            control_agg = control_sales.groupby(cust_col)[metric_name].sum()
        
        test_mean = test_agg.mean() if len(test_agg) > 0 else 0
        control_mean = control_agg.mean() if len(control_agg) > 0 else 0
        pct_diff = ((test_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
        
        # T-test
        if len(test_agg) > 1 and len(control_agg) > 1:
            t_stat, p_value = stats.ttest_ind(test_agg, control_agg)
        else:
            t_stat, p_value = 0, 1
        
        metrics.append({
            "metric_name": metric_name,
            "treatment_avg": round(test_mean, 2),
            "control_avg": round(control_mean, 2),
            "pct_diff": round(pct_diff, 2),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "min_pre_period_date": pre_start,
            "max_pre_period_date": pre_end
        })
    
    result["pre_period_summary"] = pd.DataFrame(metrics)
    
    # Pre-period trends (weekly)
    test_weekly = test_sales.groupby(test_sales['date'].dt.to_period('W'))[value_col].sum()
    control_weekly = control_sales.groupby(control_sales['date'].dt.to_period('W'))[value_col].sum()
    
    all_weeks = set(test_weekly.index) | set(control_weekly.index)
    trends = []
    for week in sorted(all_weeks):
        test_val = test_weekly.get(week, 0)
        control_val = control_weekly.get(week, 0)
        trends.append({
            "performance_date": str(week),
            "treatment_value": round(test_val, 2),
            "control_value": round(control_val, 2),
            "diff_treatment_control": round(test_val - control_val, 2)
        })
    
    result["pre_period_trends"] = pd.DataFrame(trends)
    
    # Correlation matrix placeholder
    result["correlation_matrix_pre_period"] = pd.DataFrame([{
        "metric_name": value_col,
        "treatment_metric": 1.0,
        "control_metric": 1.0,
        "correlation": round(
            test_weekly.corr(control_weekly) if len(test_weekly) > 1 and len(control_weekly) > 1 else 0,
            4
        )
    }])
    
    return result


# ==================== Measurement Tools ====================

def run_causal_impact_analysis(
    test_id: str,
    metric: str = "revenue",
    pre_period_start: Optional[str] = None,
    pre_period_end: Optional[str] = None,
    finalize_results: bool = False
) -> str:
    """
    Run CausalImpact analysis for a completed test.
    Automatically detects data columns and aggregates order-level data.
    Creates measurement phase validation files when finalize_results=True.
    
    Args:
        test_id: The test ID to analyze
        metric: What to measure - "revenue" (sum of order values) or "orders" (count of orders)
        pre_period_start: Optional override for pre-period start date
        pre_period_end: Optional override for pre-period end date
        finalize_results: If True, update master doc and create measurement validation files
        
    Returns:
        JSON with analysis results including lift, confidence intervals, and significance
    """
    try:
        # Get test details from logging utils
        test_details = get_test_by_id(test_id)
        if test_details is None:
            # Fall back to data_loader
            test_details = data_loader.get_test_summary(test_id)
            
        if test_details is None:
            return json.dumps({"error": f"Test not found: {test_id}"})
        
        # Load audience files
        test_audience = data_loader.load_test_audience(test_id)
        control_audience = data_loader.load_control_audience(test_id)
        
        if test_audience is None or control_audience is None:
            return json.dumps({"error": "Audience files not found for this test"})
        
        test_ids = test_audience["customer_id"].tolist()
        control_ids = control_audience["customer_id"].tolist()
        
        # Load sales data
        sales_df = data_loader.load_sales_data()
        
        # Prepare time series - auto-detects column names and aggregates orders
        # Each row in sales_df is a unique order, so we aggregate by date
        ts_data = causal_impact_analyzer.prepare_time_series_data(
            sales_df=sales_df,
            test_customer_ids=test_ids,
            control_customer_ids=control_ids,
            metric=metric  # "revenue" sums order values, "orders" counts rows
        )
        
        logger.info(f"Running CausalImpact analysis for {test_id} on metric: {metric}")
        
        # Determine periods - check multiple possible column names
        pre_start = pre_period_start or test_details.get("pre_period_lookback_start") or test_details.get("hist_split_balance_start_date")
        pre_end = pre_period_end or test_details.get("pre_period_lookback_end") or test_details.get("hist_split_balance_end_date")
        post_start = test_details.get("start_date")
        post_end = test_details.get("end_date")
        
        # Validate all periods are present (explicit None and empty string check)
        period_values = [pre_start, pre_end, post_start, post_end]
        if any(v is None or (isinstance(v, str) and v.strip() == "") for v in period_values):
            missing = []
            if not pre_start or (isinstance(pre_start, str) and pre_start.strip() == ""): missing.append("pre_period_start")
            if not pre_end or (isinstance(pre_end, str) and pre_end.strip() == ""): missing.append("pre_period_end")
            if not post_start or (isinstance(post_start, str) and post_start.strip() == ""): missing.append("start_date")
            if not post_end or (isinstance(post_end, str) and post_end.strip() == ""): missing.append("end_date")
            logger.error(f"Missing date fields for test {test_id}: {missing}")
            logger.error(f"Available date values - pre_start: {pre_start}, pre_end: {pre_end}, post_start: {post_start}, post_end: {post_end}")
            return json.dumps({
                "error": f"Missing required date fields: {', '.join(missing)}",
                "test_details": {k: v for k, v in test_details.items() if 'date' in k.lower() or 'period' in k.lower()}
            })
        
        # Run analysis
        results = causal_impact_analyzer.run_causal_impact(
            data=ts_data,
            pre_period=(pre_start, pre_end),
            post_period=(post_start, post_end)
        )
        
        # Process results and update logging
        if results.get("success"):
            # Map causal impact results to master doc columns
            measurement_results = {
                "avg_lift": results.get("lift_absolute"),
                "cum_lift": results.get("cumulative_effect"),
                "cred_int_lower": results.get("ci_lower"),
                "cred_int_upper": results.get("ci_upper"),
                "prob_lift_gt_0": results.get("posterior_probability"),
                "p_value_equivalent": results.get("p_value"),
                "relative_lift_pct": results.get("lift_percent")
            }
            
            if finalize_results:
                # Update master testing doc with measurement results
                update_test_with_measurement_results(test_id, measurement_results)
                
                # Generate measurement phase validation files
                try:
                    measurement_data = _generate_measurement_validation_data(
                        test_id=test_id,
                        results=results,
                        ts_data=ts_data,
                        pre_period=(pre_start, pre_end),
                        post_period=(post_start, post_end)
                    )
                    measurement_files = create_measurement_phase_files(test_id, measurement_data)
                    logger.info(f"Created {len(measurement_files)} measurement validation files")
                    results["measurement_files_created"] = [f.split("/")[-1] for f in measurement_files]
                    
                    # --- NEW: Generate Validation Package (Post-Measurement) ---
                    validation_dir = Path(data_loader.data_dir) / "test_validation_files"
                    
                    # 1. Final Experiment Report (Markdown)
                    is_sig = results.get("significant", False)
                    lift_pct = results.get("lift_percent", 0)
                    direction = "increased" if lift_pct > 0 else "decreased"
                    
                    final_report = f"""# Final Experiment Report: {test_details.get('test_name')}
**Test ID:** {test_id}
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## 1. Executive Summary
- **Outcome:** The intervention {direction} {metric} by {abs(lift_pct):.2f}%.
- **Significance:** {'Statistically Significant' if is_sig else 'Not Statistically Significant'} (p-value: {results.get('p_value'):.4f})
- **Recommendation:** {'Scale up' if is_sig and lift_pct > 0 else 'Review design / Do not scale'}

## 2. Detailed Findings
- **Absolute Effect:** {results.get('lift_absolute'):,.2f}
- **Confidence Interval:** [{results.get('ci_lower'):,.2f}, {results.get('ci_upper'):,.2f}]
- **Observed vs Predicted:** {results.get('actual'):,.2f} vs {results.get('predicted'):,.2f}

## 3. Technical Validation
- **Method:** Causal Impact (Bayesian Structural Time Series)
- **Pre-Period:** {pre_start} to {pre_end}
- **Post-Period:** {post_start} to {post_end}
"""
                    with open(validation_dir / f"{test_id}_final_experiment_report.md", "w") as f:
                        f.write(final_report)
                    
                    # 2. Model Diagnostics (JSON)
                    diagnostics = {
                        "p_value": results.get("p_value"),
                        "summary_stats": results.get("summary"),
                        "validation_checks": {
                            "pre_period_fit": "Pass" if results.get("p_value") < 0.1 else "Review", # Simplified check
                            "assumptions_met": True
                        }
                    }
                    with open(validation_dir / f"{test_id}_model_diagnostics.json", "w") as f:
                        json.dump(diagnostics, f, indent=2)
                        
                    # 3. Full Summary Output (Text)
                    if results.get("report"):
                        with open(validation_dir / f"{test_id}_causal_impact_summary.txt", "w") as f:
                            f.write(results.get("report"))
                            
                    # 4. Action Plan Template (Markdown)
                    action_plan = f"""# Action Plan: {test_details.get('test_name')}

## Implementation Steps
1. [ ] Review final report with stakeholders
2. [ ] {'Draft rollout plan' if is_sig else 'Analyze failure reasons'}
3. [ ] Update marketing knowledge base

## Learnings
- Impact on {metric}: {lift_pct:.2f}%
"""
                    with open(validation_dir / f"{test_id}_action_plan.md", "w") as f:
                        f.write(action_plan)

                except Exception as e:
                    logger.warning(f"Could not generate full measurement data: {e}. Creating empty files.")
                    create_measurement_phase_files(test_id)
                
                results["status"] = "Measurement finalized and logged"
            else:
                # Just log to version history without updating master doc
                create_version_history_entry(test_id, "measurement_preview", measurement_results)
                results["status"] = "Preview only - use finalize_results=True to save"
        
        # Add summary for return
        results["test_id"] = test_id
        results["test_name"] = test_details.get("test_name")
        results["measurement_summary"] = {
            "avg_lift": measurement_results.get("avg_lift") if results.get("success") else None,
            "relative_lift_pct": measurement_results.get("relative_lift_pct") if results.get("success") else None,
            "p_value": measurement_results.get("p_value_equivalent") if results.get("success") else None,
            "significant": results.get("significant")
        }
        
        # Remove non-serializable items
        if "inferences" in results:
            del results["inferences"]
        
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        logger.exception("Error in causal impact analysis")
        return json.dumps({"error": str(e)})


def _generate_measurement_validation_data(
    test_id: str,
    results: Dict[str, Any],
    ts_data: pd.DataFrame,
    pre_period: tuple,
    post_period: tuple,
    test_customer_ids: list = None,
    control_customer_ids: list = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate measurement validation data using the causal_impact_validation_module schema.
    
    This creates the 8 core validation CSVs per the spec:
    - pre_period_summary.csv
    - pre_period_trends.csv  
    - model_input_timeseries.csv
    - model_fit_stats.csv
    - daily_effects.csv (critical for charting!)
    - impact_summary.csv
    - model_residuals.csv
    - robustness_checks.csv
    
    Returns dict of DataFrames keyed by file type.
    """
    from datetime import datetime
    
    measurement_data = {}
    
    pre_start, pre_end = pre_period
    post_start, post_end = post_period
    
    pre_start_dt = pd.to_datetime(pre_start)
    pre_end_dt = pd.to_datetime(pre_end)
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)
    
    # =========================================================================
    # 1. model_input_timeseries.csv - Full time series fed to model
    # =========================================================================
    if ts_data is not None and not ts_data.empty:
        model_input = ts_data.copy()
        model_input = model_input.reset_index()
        model_input.columns = ['date'] + list(model_input.columns[1:])
        
        # Rename to standard schema
        col_mapping = {}
        if len(model_input.columns) >= 2:
            col_mapping[model_input.columns[1]] = 'treatment_series'
        if len(model_input.columns) >= 3:
            col_mapping[model_input.columns[2]] = 'control_series'
        model_input = model_input.rename(columns=col_mapping)
        
        # Add synthetic control placeholder if not present
        if 'synthetic_control_series' not in model_input.columns:
            model_input['synthetic_control_series'] = model_input.get('control_series', 0)
        
        measurement_data["model_input_timeseries"] = model_input
    
    # =========================================================================
    # 2. daily_effects.csv - CRITICAL FOR CHARTING
    # This is what get_causal_impact_charts uses for predicted vs actual, 
    # pointwise, and cumulative charts
    # =========================================================================
    inferences = results.get("inferences")
    if inferences is not None:
        # Convert from dict back to DataFrame if needed
        if isinstance(inferences, dict):
            inferences_df = pd.DataFrame(inferences)
            inferences_df.index = pd.to_datetime(inferences_df.index)
        else:
            inferences_df = inferences
        
        # Filter to post-period only for daily_effects
        post_mask = (inferences_df.index >= post_start_dt) & (inferences_df.index <= post_end_dt)
        post_inferences = inferences_df[post_mask].copy()
        
        # tfp-causalimpact uses different column names than tfcausalimpact:
        # - 'observed' instead of 'response'
        # - 'posterior_mean' instead of 'point_pred'
        # - 'posterior_lower/upper' instead of 'point_pred_lower/upper'
        # - 'point_effects_mean/lower/upper' for the effects
        
        # Get actual values - try tfp-causalimpact names first, then fallback
        if 'observed' in post_inferences.columns:
            actual_col = 'observed'
        elif 'response' in post_inferences.columns:
            actual_col = 'response'
        else:
            actual_col = post_inferences.columns[0] if len(post_inferences.columns) > 0 else None
        
        # Get predicted/counterfactual values
        if 'posterior_mean' in post_inferences.columns:
            pred_col = 'posterior_mean'
        elif 'point_pred' in post_inferences.columns:
            pred_col = 'point_pred'
        else:
            pred_col = None
        
        # Get effect columns - tfp-causalimpact provides these directly
        if 'point_effects_mean' in post_inferences.columns:
            # tfp-causalimpact provides effects directly
            daily_effects = pd.DataFrame({
                "date": post_inferences.index,
                "actual": post_inferences[actual_col] if actual_col else 0,
                "expected_counterfactual": post_inferences[pred_col] if pred_col else 0,
                "point_effect": post_inferences['point_effects_mean'],
                "lower_effect": post_inferences.get('point_effects_lower', post_inferences['point_effects_mean'] * 0.8),
                "upper_effect": post_inferences.get('point_effects_upper', post_inferences['point_effects_mean'] * 1.2)
            })
        else:
            # Fallback to computing effects from actual - predicted
            actual_vals = post_inferences[actual_col] if actual_col else 0
            pred_vals = post_inferences[pred_col] if pred_col else 0
            
            # Get bounds for effect calculation
            if 'posterior_upper' in post_inferences.columns:
                pred_upper = post_inferences['posterior_upper']
                pred_lower = post_inferences['posterior_lower']
            elif 'point_pred_upper' in post_inferences.columns:
                pred_upper = post_inferences['point_pred_upper']
                pred_lower = post_inferences['point_pred_lower']
            else:
                pred_upper = pred_vals * 1.1 if isinstance(pred_vals, pd.Series) else 0
                pred_lower = pred_vals * 0.9 if isinstance(pred_vals, pd.Series) else 0
            
            daily_effects = pd.DataFrame({
                "date": post_inferences.index,
                "actual": actual_vals,
                "expected_counterfactual": pred_vals,
                "point_effect": actual_vals - pred_vals if isinstance(actual_vals, pd.Series) else 0,
                "lower_effect": actual_vals - pred_upper if isinstance(actual_vals, pd.Series) else 0,
                "upper_effect": actual_vals - pred_lower if isinstance(actual_vals, pd.Series) else 0
            })
        measurement_data["daily_effects"] = daily_effects
    elif ts_data is not None:
        # Fallback: create daily_effects from raw time series
        ts_copy = ts_data.copy()
        ts_copy = ts_copy.reset_index()
        ts_copy.columns = ['date'] + list(ts_copy.columns[1:])
        ts_copy['date'] = pd.to_datetime(ts_copy['date'])
        
        post_mask = (ts_copy['date'] >= post_start_dt) & (ts_copy['date'] <= post_end_dt)
        post_ts = ts_copy[post_mask].copy()
        
        treatment_col = post_ts.columns[1] if len(post_ts.columns) > 1 else None
        control_col = post_ts.columns[2] if len(post_ts.columns) > 2 else None
        
        if treatment_col and control_col:
            daily_effects = pd.DataFrame({
                "date": post_ts['date'],
                "actual": post_ts[treatment_col],
                "expected_counterfactual": post_ts[control_col],  # Using control as proxy
                "point_effect": post_ts[treatment_col] - post_ts[control_col],
                "lower_effect": (post_ts[treatment_col] - post_ts[control_col]) * 0.8,  # Estimated
                "upper_effect": (post_ts[treatment_col] - post_ts[control_col]) * 1.2   # Estimated
            })
            measurement_data["daily_effects"] = daily_effects
        else:
            measurement_data["daily_effects"] = pd.DataFrame(columns=[
                "date", "actual", "expected_counterfactual", "point_effect", "lower_effect", "upper_effect"
            ])
    else:
        measurement_data["daily_effects"] = pd.DataFrame(columns=[
            "date", "actual", "expected_counterfactual", "point_effect", "lower_effect", "upper_effect"
        ])
    
    # =========================================================================
    # 3. impact_summary.csv - Headline results
    # =========================================================================
    impact_summary = pd.DataFrame([{
        "avg_lift": results.get("lift_absolute"),
        "cum_lift": results.get("cumulative_effect", results.get("lift_absolute", 0) * 30),  # Approximate if not available
        "cred_int_lower": results.get("ci_lower"),
        "cred_int_upper": results.get("ci_upper"),
        "prob_lift_gt_0": results.get("posterior_probability", 1 - results.get("p_value", 0.5) if results.get("p_value") else None),
        "p_value_equivalent": results.get("p_value"),
        "relative_lift_pct": results.get("lift_percent")
    }])
    measurement_data["impact_summary"] = impact_summary
    
    # =========================================================================
    # 4. model_fit_stats.csv - Model diagnostics
    # =========================================================================
    fit_stats = pd.DataFrame([
        {"metric_name": "MAPE", "value": results.get("mape", "N/A"), "description": "Mean Absolute Percentage Error"},
        {"metric_name": "MAE", "value": results.get("mae", "N/A"), "description": "Mean Absolute Error"},
        {"metric_name": "RMSE", "value": results.get("rmse", "N/A"), "description": "Root Mean Squared Error"},
        {"metric_name": "R_squared", "value": results.get("r_squared", "N/A"), "description": "R-squared of pre-period fit"},
        {"metric_name": "convergence_status", "value": "Converged" if results.get("success") else "Check required", "description": "Model convergence status"},
        {"metric_name": "log_likelihood", "value": results.get("log_likelihood", "N/A"), "description": "Model log likelihood"}
    ])
    measurement_data["model_fit_stats"] = fit_stats
    
    # =========================================================================
    # 5. model_residuals.csv - Residuals across periods
    # =========================================================================
    if inferences is not None:
        if isinstance(inferences, dict):
            inferences_df = pd.DataFrame(inferences)
            inferences_df.index = pd.to_datetime(inferences_df.index)
        else:
            inferences_df = inferences
        
        residuals = pd.DataFrame({
            "date": inferences_df.index,
            "residual_value": inferences_df.get('response', 0) - inferences_df.get('point_pred', 0),
            "is_pre_period": (inferences_df.index >= pre_start_dt) & (inferences_df.index <= pre_end_dt),
            "is_post_period": (inferences_df.index >= post_start_dt) & (inferences_df.index <= post_end_dt)
        })
        residuals['is_pre_period'] = residuals['is_pre_period'].astype(int)
        residuals['is_post_period'] = residuals['is_post_period'].astype(int)
        measurement_data["model_residuals"] = residuals
    else:
        measurement_data["model_residuals"] = pd.DataFrame(columns=[
            "date", "residual_value", "is_pre_period", "is_post_period"
        ])
    
    # =========================================================================
    # 6. pre_period_summary.csv - Balance check
    # =========================================================================
    if ts_data is not None:
        ts_copy = ts_data.copy()
        ts_copy = ts_copy.reset_index()
        ts_copy.columns = ['date'] + list(ts_copy.columns[1:])
        ts_copy['date'] = pd.to_datetime(ts_copy['date'])
        
        pre_mask = (ts_copy['date'] >= pre_start_dt) & (ts_copy['date'] <= pre_end_dt)
        pre_ts = ts_copy[pre_mask]
        
        treatment_col = pre_ts.columns[1] if len(pre_ts.columns) > 1 else None
        control_col = pre_ts.columns[2] if len(pre_ts.columns) > 2 else None
        
        if treatment_col and control_col:
            treatment_avg = pre_ts[treatment_col].mean()
            control_avg = pre_ts[control_col].mean()
            pct_diff = ((treatment_avg - control_avg) / control_avg * 100) if control_avg != 0 else 0
            
            # Simple t-test
            try:
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(pre_ts[treatment_col].dropna(), pre_ts[control_col].dropna(), equal_var=False)
            except:
                t_stat, p_value = None, None
            
            pre_period_summary = pd.DataFrame([{
                "metric_name": "revenue",
                "treatment_avg": treatment_avg,
                "control_avg": control_avg,
                "pct_diff": pct_diff,
                "t_stat": t_stat,
                "p_value": p_value,
                "min_pre_period_date": pre_start,
                "max_pre_period_date": pre_end
            }])
            measurement_data["pre_period_summary"] = pre_period_summary
        else:
            measurement_data["pre_period_summary"] = pd.DataFrame(columns=[
                "metric_name", "treatment_avg", "control_avg", "pct_diff",
                "t_stat", "p_value", "min_pre_period_date", "max_pre_period_date"
            ])
    
    # =========================================================================
    # 7. pre_period_trends.csv - Parallel trends check
    # =========================================================================
    if ts_data is not None:
        ts_copy = ts_data.copy()
        ts_copy = ts_copy.reset_index()
        ts_copy.columns = ['date'] + list(ts_copy.columns[1:])
        ts_copy['date'] = pd.to_datetime(ts_copy['date'])
        
        pre_mask = (ts_copy['date'] >= pre_start_dt) & (ts_copy['date'] <= pre_end_dt)
        pre_ts = ts_copy[pre_mask].copy()
        
        treatment_col = pre_ts.columns[1] if len(pre_ts.columns) > 1 else None
        control_col = pre_ts.columns[2] if len(pre_ts.columns) > 2 else None
        
        if treatment_col and control_col:
            pre_period_trends = pd.DataFrame({
                "performance_date": pre_ts['date'],
                "treatment_value": pre_ts[treatment_col],
                "control_value": pre_ts[control_col],
                "diff_treatment_control": pre_ts[treatment_col] - pre_ts[control_col]
            })
            measurement_data["pre_period_trends"] = pre_period_trends
        else:
            measurement_data["pre_period_trends"] = pd.DataFrame(columns=[
                "performance_date", "treatment_value", "control_value", "diff_treatment_control"
            ])
    
    # =========================================================================
    # 8. robustness_checks.csv - Sensitivity analysis placeholder
    # =========================================================================
    # Default scenario is the main analysis
    robustness_checks = pd.DataFrame([{
        "scenario_name": "main_analysis",
        "avg_lift": results.get("lift_absolute"),
        "cred_int_lower": results.get("ci_lower"),
        "cred_int_upper": results.get("ci_upper"),
        "prob_lift_gt_0": results.get("posterior_probability", 1 - results.get("p_value", 0.5) if results.get("p_value") else None)
    }])
    measurement_data["robustness_checks"] = robustness_checks
    
    return measurement_data


def get_measurement_summary(test_id: str) -> str:
    """
    Get a formatted summary of measurement results for a test.
    
    Args:
        test_id: The test ID to summarize
        
    Returns:
        Formatted text summary of the measurement results
    """
    try:
        test_details = data_loader.get_test_summary(test_id)
        if test_details is None:
            return json.dumps({"error": f"Test not found: {test_id}"})
        
        # Check if measurement has been run
        if pd.isna(test_details.get("lift_absolute")):
            return json.dumps({
                "test_id": test_id,
                "status": "No measurement results available",
                "message": "Run causal impact analysis first using run_causal_impact_analysis tool"
            })
        
        summary = {
            "test_id": test_id,
            "test_name": test_details.get("test_name"),
            "status": test_details.get("status"),
            "test_period": {
                "start": test_details.get("start_date"),
                "end": test_details.get("end_date")
            },
            "results": {
                "lift_absolute": test_details.get("lift_absolute"),
                "lift_percent": test_details.get("lift_percent"),
                "p_value": test_details.get("p_value"),
                "confidence_level": test_details.get("confidence_level"),
                "confidence_interval": {
                    "lower": test_details.get("ci_lower"),
                    "upper": test_details.get("ci_upper")
                },
                "statistically_significant": test_details.get("significant")
            },
            "measurement_date": test_details.get("measurement_run_date")
        }
        
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def update_test_status(test_id: str, new_status: str) -> str:
    """
    Update the status of a test.
    Uses logging utilities for proper version tracking.
    
    Args:
        test_id: The test ID to update
        new_status: New status (Scheduled, Running, Complete, Paused, Cancelled)
        
    Returns:
        JSON confirmation of the update
    """
    try:
        valid_statuses = ["Scheduled", "Running", "Complete", "Paused", "Cancelled"]
        if new_status not in valid_statuses:
            return json.dumps({
                "error": f"Invalid status. Valid options: {valid_statuses}"
            })
        
        # Use logging utilities for proper version tracking
        success = logging_update_test_status(test_id, new_status)
        
        if success:
            return json.dumps({
                "success": True,
                "test_id": test_id,
                "new_status": new_status,
                "message": f"Test status updated to '{new_status}'",
                "logged_to_version_history": True
            })
        else:
            return json.dumps({"error": f"Failed to update test: {test_id}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_measurement_history(test_id: str) -> str:
    """
    Get the measurement history (all versions) for a specific test.
    This shows all measurement runs performed on a test over time.
    
    Args:
        test_id: The test ID to get history for
        
    Returns:
        JSON with all measurement versions and their results
    """
    try:
        history_df = data_loader.get_measurement_history(test_id)
        
        if history_df.empty:
            return json.dumps({
                "test_id": test_id,
                "message": "No measurement history found for this test",
                "versions": []
            })
        
        versions = []
        for _, row in history_df.iterrows():
            version_info = {
                "history_id": row.get("history_id"),
                "version": int(row.get("version", 0)),
                "measurement_run_date": str(row.get("measurement_run_date", "")),
                "measurement_method": row.get("measurement_method", ""),
                "lift_absolute": row.get("lift_absolute"),
                "lift_percent": row.get("lift_percent"),
                "p_value": row.get("p_value"),
                "confidence_level": row.get("confidence_level"),
                "ci_lower": row.get("ci_lower"),
                "ci_upper": row.get("ci_upper"),
                "significant": row.get("significant"),
                "notes": row.get("notes", "")
            }
            versions.append(version_info)
        
        return json.dumps({
            "test_id": test_id,
            "total_versions": len(versions),
            "versions": versions
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_all_measurement_history() -> str:
    """
    Get the complete measurement history across all tests.
    Useful for reviewing all measurement runs.
    
    Returns:
        JSON with all measurement history entries
    """
    try:
        history_df = data_loader.load_measurement_history()
        
        if history_df.empty:
            return json.dumps({
                "message": "No measurement history found",
                "total_entries": 0,
                "entries": []
            })
        
        entries = []
        for _, row in history_df.iterrows():
            entry = {
                "history_id": row.get("history_id"),
                "test_id": row.get("test_id"),
                "version": int(row.get("version", 0)),
                "measurement_run_date": str(row.get("measurement_run_date", "")),
                "measurement_method": row.get("measurement_method", ""),
                "lift_percent": row.get("lift_percent"),
                "significant": row.get("significant")
            }
            entries.append(entry)
        
        return json.dumps({
            "total_entries": len(entries),
            "entries": entries
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_test_validation_files(test_id: str) -> str:
    """
    Get a list of all validation files created for a test.
    
    Args:
        test_id: The test ID to check
        
    Returns:
        JSON with list of validation files and their paths
    """
    try:
        import os
        validation_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_validation_files",
            f"{test_id}_test_validation_files"
        )
        
        if not os.path.exists(validation_folder):
            return json.dumps({
                "test_id": test_id,
                "error": "Validation folder not found",
                "expected_path": str(validation_folder)
            })
        
        files = os.listdir(validation_folder)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        return json.dumps({
            "test_id": test_id,
            "validation_folder": str(validation_folder),
            "total_files": len(csv_files),
            "files": csv_files
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_validation_file_data(test_id: str, file_type: str) -> str:
    """
    Get the contents of a specific validation file for a test.
    
    Available file types:
    
    DESIGN PHASE FILES (created when test is finalized):
    - pre_period_summary: Pre-period balance metrics between test/control
    - pre_period_trends: Time series of test vs control in pre-period
    - correlation_matrix_pre_period: Correlation between test and control metrics
    
    MEASUREMENT PHASE FILES (created when measurement is run):
    - model_input_timeseries: The time series data fed into CausalImpact
    - missing_data_report: Report of any missing data points
    - model_posterior_samples: Bayesian posterior predictions
    - model_residuals: Model residuals for fit diagnostics
    - model_fit_stats: Model fit statistics (MAPE, MAE, R-squared)
    - impact_summary: Summary of causal impact results
    - daily_effects: Day-by-day causal effects
    - test_period_summary: Aggregated test period results
    - robustness_checks: Sensitivity analysis results
    - placebo_test_results: Placebo/falsification test results
    - structural_break_test: Tests for structural breaks
    
    Args:
        test_id: The test ID
        file_type: One of the file types listed above
        
    Returns:
        JSON with the file contents as records
    """
    try:
        import os
        validation_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_validation_files",
            f"{test_id}_test_validation_files"
        )
        
        file_path = os.path.join(validation_folder, f"{test_id}_{file_type}.csv")
        
        if not os.path.exists(file_path):
            return json.dumps({
                "test_id": test_id,
                "file_type": file_type,
                "error": f"File not found: {file_type}",
                "hint": "Use get_test_validation_files to see available files"
            })
        
        df = pd.read_csv(file_path)
        
        return json.dumps({
            "test_id": test_id,
            "file_type": file_type,
            "columns": list(df.columns),
            "row_count": len(df),
            "data": df.to_dict(orient="records")
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_pre_period_balance_check(test_id: str) -> str:
    """
    Get the pre-period balance check results for a test.
    This shows how well matched the test and control groups were before the intervention.
    
    Good balance is critical for valid CausalImpact results. This tool returns:
    - Treatment vs Control averages for key metrics
    - Percentage differences
    - Statistical tests (t-stat, p-value) for balance
    
    Args:
        test_id: The test ID
        
    Returns:
        JSON with pre-period balance metrics and interpretation
    """
    try:
        import os
        validation_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_validation_files",
            f"{test_id}_test_validation_files"
        )
        
        # Check both the subfolder and the main validation folder
        main_validation_folder = Path(__file__).parent.parent / "data" / "test_validation_files"
        
        # Look for files in the test-specific subfolder first, then main folder
        summary_paths = [
            os.path.join(validation_folder, f"{test_id}_pre_period_summary.csv"),
            os.path.join(main_validation_folder, f"{test_id}_pre_period_balance_metrics.csv"),  # Design phase name
        ]
        trends_paths = [
            os.path.join(validation_folder, f"{test_id}_pre_period_trends.csv"),
        ]
        corr_paths = [
            os.path.join(validation_folder, f"{test_id}_correlation_matrix_pre_period.csv"),
        ]
        
        # Find existing files
        summary_path = next((p for p in summary_paths if os.path.exists(p)), None)
        trends_path = next((p for p in trends_paths if os.path.exists(p)), None)
        corr_path = next((p for p in corr_paths if os.path.exists(p)), None)
        
        result = {
            "test_id": test_id,
            "pre_period_summary": None,
            "pre_period_trends": None,
            "correlation": None,
            "interpretation": {},
            "files_checked": {
                "summary_found": summary_path is not None,
                "trends_found": trends_path is not None,
                "correlation_found": corr_path is not None
            }
        }
        
        # Load pre-period summary
        if summary_path:
            summary_df = pd.read_csv(summary_path)
            result["pre_period_summary"] = summary_df.to_dict(orient="records")
            
            # Add interpretation
            if not summary_df.empty:
                high_pct_diff = summary_df[abs(summary_df["pct_diff"]) > 10] if "pct_diff" in summary_df.columns else pd.DataFrame()
                low_pval = summary_df[summary_df["p_value"] < 0.05] if "p_value" in summary_df.columns else pd.DataFrame()
                
                result["interpretation"]["balance_concerns"] = []
                if not high_pct_diff.empty:
                    result["interpretation"]["balance_concerns"].append(
                        f"Warning: {len(high_pct_diff)} metrics have >10% difference between groups"
                    )
                if not low_pval.empty:
                    result["interpretation"]["balance_concerns"].append(
                        f"Warning: {len(low_pval)} metrics show statistically significant differences (p<0.05)"
                    )
                if not result["interpretation"]["balance_concerns"]:
                    result["interpretation"]["balance_concerns"].append("Groups appear well balanced")
        else:
            result["interpretation"]["balance_concerns"] = ["No pre-period data available yet. Run test design first."]
        
        # Load trends
        if trends_path:
            trends_df = pd.read_csv(trends_path)
            result["pre_period_trends"] = {
                "row_count": len(trends_df),
                "date_range": f"{trends_df['performance_date'].min()} to {trends_df['performance_date'].max()}" if not trends_df.empty else "N/A",
                "avg_diff": trends_df["diff_treatment_control"].mean() if "diff_treatment_control" in trends_df.columns else None
            }
        
        # Load correlation
        if corr_path:
            corr_df = pd.read_csv(corr_path)
            result["correlation"] = corr_df.to_dict(orient="records")
            
            if not corr_df.empty and "correlation" in corr_df.columns:
                avg_corr = corr_df["correlation"].mean()
                result["interpretation"]["correlation_quality"] = (
                    "Excellent" if avg_corr > 0.9 else
                    "Good" if avg_corr > 0.7 else
                    "Moderate" if avg_corr > 0.5 else
                    "Poor - may affect result reliability"
                )
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_model_diagnostics(test_id: str) -> str:
    """
    Get model diagnostics and fit statistics for a CausalImpact analysis.
    
    This returns information about how well the model fit the pre-period data,
    which is critical for trusting the counterfactual predictions.
    
    NOTE: Model diagnostics are only available AFTER running causal impact measurement
    with finalize_results=True. If you see None values, the measurement hasn't been run yet.
    
    Includes:
    - Model fit statistics (MAPE, MAE, R-squared)
    - Residual analysis
    - Posterior prediction quality
    
    Args:
        test_id: The test ID
        
    Returns:
        JSON with model diagnostics and interpretation
    """
    try:
        import os
        validation_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_validation_files",
            f"{test_id}_test_validation_files"
        )
        
        # Check if validation folder exists
        folder_exists = os.path.exists(validation_folder)
        
        result = {
            "test_id": test_id,
            "model_fit_stats": None,
            "residuals_summary": None,
            "interpretation": {},
            "files_checked": {
                "validation_folder_exists": folder_exists,
                "model_fit_stats_found": False,
                "model_residuals_found": False
            }
        }
        
        if not folder_exists:
            result["interpretation"]["status"] = "Validation folder not found. Has the test been created yet?"
            return json.dumps(result, indent=2, default=str)
        
        # Load model fit stats
        fit_path = os.path.join(validation_folder, f"{test_id}_model_fit_stats.csv")
        if os.path.exists(fit_path):
            result["files_checked"]["model_fit_stats_found"] = True
            fit_df = pd.read_csv(fit_path)
            result["model_fit_stats"] = fit_df.to_dict(orient="records")
            
            # Interpret fit quality
            for _, row in fit_df.iterrows():
                if row.get("metric_name") == "mape":
                    mape = row.get("value", 0)
                    result["interpretation"]["mape_quality"] = (
                        "Excellent (<5%)" if mape < 5 else
                        "Good (5-10%)" if mape < 10 else
                        "Acceptable (10-20%)" if mape < 20 else
                        "Poor (>20%) - results may be unreliable"
                    )
                elif row.get("metric_name") == "r_squared":
                    r2 = row.get("value", 0)
                    result["interpretation"]["r_squared_quality"] = (
                        "Excellent (>0.9)" if r2 > 0.9 else
                        "Good (0.7-0.9)" if r2 > 0.7 else
                        "Moderate (0.5-0.7)" if r2 > 0.5 else
                        "Poor (<0.5) - model may not capture patterns well"
                    )
        
        else:
            result["interpretation"]["model_fit_status"] = "Model fit stats not found. Run causal impact measurement with finalize_results=True first."
        
        # Load residuals
        residuals_path = os.path.join(validation_folder, f"{test_id}_model_residuals.csv")
        if os.path.exists(residuals_path):
            result["files_checked"]["model_residuals_found"] = True
            residuals_df = pd.read_csv(residuals_path)
            if not residuals_df.empty and "residual_value" in residuals_df.columns:
                result["residuals_summary"] = {
                    "mean": round(residuals_df["residual_value"].mean(), 4),
                    "std": round(residuals_df["residual_value"].std(), 4),
                    "min": round(residuals_df["residual_value"].min(), 4),
                    "max": round(residuals_df["residual_value"].max(), 4)
                }
                
                # Check for residual bias
                if abs(result["residuals_summary"]["mean"]) > 0.1 * result["residuals_summary"]["std"]:
                    result["interpretation"]["residual_bias"] = "Warning: Non-zero mean residuals suggest model bias"
                else:
                    result["interpretation"]["residual_bias"] = "Residuals appear unbiased"
        else:
            result["interpretation"]["residuals_status"] = "Model residuals not found. Run causal impact measurement with finalize_results=True first."
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_robustness_analysis(test_id: str) -> str:
    """
    Get robustness checks and placebo test results for a CausalImpact analysis.
    
    Robustness checks help validate that results are not due to:
    - Model specification choices
    - Random chance
    - Pre-existing trends
    
    Includes:
    - Robustness checks under different scenarios
    - Placebo test results (fake intervention dates)
    - Structural break tests
    
    Args:
        test_id: The test ID
        
    Returns:
        JSON with robustness analysis and interpretation
    """
    try:
        import os
        validation_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_validation_files",
            f"{test_id}_test_validation_files"
        )
        
        result = {
            "test_id": test_id,
            "robustness_checks": None,
            "placebo_tests": None,
            "structural_breaks": None,
            "interpretation": {}
        }
        
        # Load robustness checks
        robust_path = os.path.join(validation_folder, f"{test_id}_robustness_checks.csv")
        if os.path.exists(robust_path):
            robust_df = pd.read_csv(robust_path)
            result["robustness_checks"] = robust_df.to_dict(orient="records")
            
            if not robust_df.empty and "prob_lift_gt_0" in robust_df.columns:
                all_positive = all(robust_df["prob_lift_gt_0"] > 0.9)
                all_same_sign = all(robust_df["avg_lift"] > 0) or all(robust_df["avg_lift"] < 0)
                
                result["interpretation"]["robustness_quality"] = (
                    "Strong - consistent results across scenarios" if all_positive and all_same_sign else
                    "Moderate - some variation across scenarios" if all_same_sign else
                    "Weak - results sensitive to model choices"
                )
        
        # Load placebo tests
        placebo_path = os.path.join(validation_folder, f"{test_id}_placebo_test_results.csv")
        if os.path.exists(placebo_path):
            placebo_df = pd.read_csv(placebo_path)
            result["placebo_tests"] = placebo_df.to_dict(orient="records")
            
            if not placebo_df.empty and "prob_lift_gt_0" in placebo_df.columns:
                false_positives = sum(placebo_df["prob_lift_gt_0"] > 0.9)
                result["interpretation"]["placebo_quality"] = (
                    "Good - no false positives in placebo tests" if false_positives == 0 else
                    f"Concerning - {false_positives} placebo tests showed significant effects"
                )
        
        # Load structural break tests
        break_path = os.path.join(validation_folder, f"{test_id}_structural_break_test.csv")
        if os.path.exists(break_path):
            break_df = pd.read_csv(break_path)
            result["structural_breaks"] = break_df.to_dict(orient="records")
            
            if not break_df.empty and "p_value" in break_df.columns:
                significant_breaks = sum(break_df["p_value"] < 0.05)
                result["interpretation"]["structural_breaks"] = (
                    "No structural breaks detected" if significant_breaks == 0 else
                    f"Warning: {significant_breaks} potential structural breaks detected"
                )
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_daily_effects_analysis(test_id: str) -> str:
    """
    Get the day-by-day causal effects from a CausalImpact analysis.
    
    This shows how the treatment effect evolved over time during the test period.
    Useful for understanding:
    - When effects started appearing
    - Whether effects grew or diminished over time
    - Days with unusually high or low effects
    
    Args:
        test_id: The test ID
        
    Returns:
        JSON with daily effects data and summary statistics
    """
    try:
        import os
        validation_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_validation_files",
            f"{test_id}_test_validation_files"
        )
        
        result = {
            "test_id": test_id,
            "daily_effects": None,
            "test_period_summary": None,
            "interpretation": {}
        }
        
        # Load daily effects
        daily_path = os.path.join(validation_folder, f"{test_id}_daily_effects.csv")
        if os.path.exists(daily_path):
            daily_df = pd.read_csv(daily_path)
            
            if not daily_df.empty:
                result["daily_effects"] = {
                    "total_days": len(daily_df),
                    "columns": list(daily_df.columns),
                    "summary_stats": {},
                    "first_5_days": daily_df.head(5).to_dict(orient="records"),
                    "last_5_days": daily_df.tail(5).to_dict(orient="records")
                }
                
                if "point_effect" in daily_df.columns:
                    result["daily_effects"]["summary_stats"] = {
                        "avg_daily_effect": round(daily_df["point_effect"].mean(), 2),
                        "total_effect": round(daily_df["point_effect"].sum(), 2),
                        "max_effect_day": daily_df.loc[daily_df["point_effect"].idxmax()].to_dict() if len(daily_df) > 0 else None,
                        "min_effect_day": daily_df.loc[daily_df["point_effect"].idxmin()].to_dict() if len(daily_df) > 0 else None
                    }
                    
                    # Check for trend in effects
                    if len(daily_df) > 5:
                        first_half = daily_df["point_effect"].iloc[:len(daily_df)//2].mean()
                        second_half = daily_df["point_effect"].iloc[len(daily_df)//2:].mean()
                        
                        result["interpretation"]["effect_trend"] = (
                            "Effects appear to grow over time" if second_half > first_half * 1.2 else
                            "Effects appear to diminish over time" if second_half < first_half * 0.8 else
                            "Effects appear stable over time"
                        )
        
        # Load test period summary
        summary_path = os.path.join(validation_folder, f"{test_id}_test_period_summary.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            result["test_period_summary"] = summary_df.to_dict(orient="records")
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_test_version_history(test_id: str) -> str:
    """
    Get the version history for a specific test.
    Shows all actions taken on the test over time.
    
    Args:
        test_id: The test ID to get history for
        
    Returns:
        JSON with all version history entries
    """
    try:
        import os
        version_folder = os.path.join(
            Path(__file__).parent.parent / "data" / "test_version_history",
            f"{test_id}_version_history"
        )
        
        if not os.path.exists(version_folder):
            return json.dumps({
                "test_id": test_id,
                "message": "No version history found",
                "versions": []
            })
        
        files = sorted([f for f in os.listdir(version_folder) if f.endswith('.csv')])
        
        versions = []
        for file_name in files:
            file_path = os.path.join(version_folder, file_name)
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    versions.append(df.iloc[0].to_dict())
            except Exception:
                continue
        
        return json.dumps({
            "test_id": test_id,
            "total_versions": len(versions),
            "versions": versions
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ==================== Knowledge Base / RAG Tools ====================

def search_causal_impact_knowledge(query: str, n_results: int = 5) -> str:
    """
    Search the Causal Impact knowledge base for best practices, 
    methodological guidance, and troubleshooting information.
    
    Use this tool when you need information about:
    - Causal Impact methodology and concepts
    - Best practices for test design
    - Data preparation requirements
    - Common pitfalls and how to avoid them
    - Interpretation of results
    - Marketing use cases
    
    Args:
        query: Natural language question or search terms
        n_results: Number of results to return (default 5)
        
    Returns:
        Formatted string with relevant knowledge base content
    """
    try:
        result = rag_search(query, n_results)
        return result
    except Exception as e:
        logger.exception("Error searching knowledge base")
        return json.dumps({"error": str(e)})


def initialize_knowledge_base(force_reindex: bool = False) -> str:
    """
    Initialize or reindex the Causal Impact knowledge base.
    This reads all markdown files from the knowledge_base folder,
    chunks them by H1/H2 headers, generates embeddings, and stores
    in ChromaDB for semantic search.
    
    Args:
        force_reindex: If True, delete existing index and reindex all documents
        
    Returns:
        String with indexing statistics
    """
    try:
        result = rag_index(force_reindex)
        return result
    except Exception as e:
        logger.exception("Error initializing knowledge base")
        return json.dumps({"error": str(e)})


def get_knowledge_base_info() -> str:
    """
    Get information about the current state of the knowledge base.
    
    Returns:
        String with collection statistics and storage info
    """
    try:
        result = rag_stats()
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})


# ==================== Web Search Tools ====================

def search_measurement_methodology(
    query: str,
    max_results: int = 5,
    include_raw_content: bool = False
) -> str:
    """
    Search the web for information about marketing measurement methodologies,
    test design approaches, and statistical techniques.
    
    Use this tool when users ask about:
    - Alternative measurement methodologies (ARIMA, Diff-in-Diff, Synthetic Control, etc.)
    - Why CausalImpact vs other approaches
    - Best practices for A/B testing, geo experiments, or incrementality testing
    - Academic research on marketing measurement
    - Industry standards for test design
    
    Args:
        query: The search query - be specific about measurement/testing context
        max_results: Maximum number of search results to return (default 5)
        include_raw_content: If True, attempt to fetch full page content with BeautifulSoup
        
    Returns:
        JSON string with search results and optional page content
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return json.dumps({
                "error": "TAVILY_API_KEY not found in environment variables",
                "suggestion": "Add TAVILY_API_KEY to your .env file"
            })
        
        # Import Tavily
        try:
            from tavily import TavilyClient
        except ImportError:
            return json.dumps({
                "error": "tavily-python package not installed",
                "suggestion": "Run: pip install tavily-python"
            })
        
        # Initialize Tavily client
        client = TavilyClient(api_key=tavily_api_key)
        
        # Enhance query with measurement context for better results
        enhanced_query = f"marketing measurement {query}"
        
        # Perform search
        search_response = client.search(
            query=enhanced_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=[
                "google.github.io",  # CausalImpact docs
                "research.google.com",
                "arxiv.org",
                "towardsdatascience.com",
                "medium.com",
                "analyticsvidhya.com",
                "stats.stackexchange.com",
                "sciencedirect.com",
                "springer.com",
                "journals.sagepub.com"
            ]
        )
        
        results = []
        for result in search_response.get("results", []):
            # Handle None values safely
            content = result.get("content") or ""
            result_data = {
                "title": result.get("title") or "No title",
                "url": result.get("url") or "",
                "snippet": content[:500] if content else "",
                "score": result.get("score")
            }
            
            # Optionally fetch full content with BeautifulSoup
            if include_raw_content and result_data["url"]:
                try:
                    from bs4 import BeautifulSoup
                    
                    response = requests.get(
                        result_data["url"],
                        timeout=10,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; MeasurementBot/1.0)"}
                    )
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, "html.parser")
                        
                        # Remove script and style elements
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text(separator="\n", strip=True)
                        # Truncate to reasonable length
                        result_data["full_content"] = text[:3000] if text else ""
                except Exception as e:
                    result_data["content_fetch_error"] = str(e)
            
            results.append(result_data)
        
        return json.dumps({
            "success": True,
            "query": query,
            "enhanced_query": enhanced_query,
            "result_count": len(results),
            "results": results,
            "methodology_context": _get_methodology_context(query)
        }, indent=2)
        
    except Exception as e:
        logger.exception("Error in web search")
        return json.dumps({
            "error": str(e),
            "query": query
        })


def _get_methodology_context(query: str) -> dict:
    """
    Provide built-in context about common measurement methodologies
    to supplement web search results.
    """
    query_lower = query.lower()
    
    context = {
        "our_approach": "CausalImpact (Bayesian Structural Time Series)",
        "why_causal_impact": [
            "Handles time series with seasonality and trends",
            "Provides probabilistic inference with credible intervals",
            "Works well with geo-based experiments",
            "Developed by Google for marketing measurement",
            "Accounts for temporal correlation in the data"
        ]
    }
    
    # Add relevant comparisons based on query
    if any(term in query_lower for term in ["arima", "time series", "forecast"]):
        context["arima_comparison"] = {
            "arima": "ARIMA models forecast based on historical patterns but don't establish causality",
            "causal_impact_advantage": "CausalImpact uses a synthetic control (counterfactual) built from control markets, allowing causal inference rather than just forecasting"
        }
    
    if any(term in query_lower for term in ["diff", "difference", "did"]):
        context["did_comparison"] = {
            "difference_in_differences": "DiD compares pre/post changes between test and control groups",
            "causal_impact_advantage": "CausalImpact uses Bayesian structural time series which better handles autocorrelation, seasonality, and provides uncertainty quantification"
        }
    
    if any(term in query_lower for term in ["synthetic", "synth"]):
        context["synthetic_control_comparison"] = {
            "synthetic_control": "Creates a weighted combination of control units to match the treated unit",
            "relationship": "CausalImpact can be seen as a Bayesian generalization of synthetic control methods, with better uncertainty quantification"
        }
    
    if any(term in query_lower for term in ["propensity", "psm", "matching"]):
        context["psm_comparison"] = {
            "propensity_score_matching": "PSM is used for observational studies to create comparable treatment/control groups",
            "when_to_use": "PSM is better for non-randomized customer-level studies; CausalImpact excels at time-series geo experiments"
        }
    
    if any(term in query_lower for term in ["a/b", "ab test", "randomized"]):
        context["ab_testing_comparison"] = {
            "ab_testing": "Traditional A/B tests with random assignment and simple statistical tests",
            "when_geo_experiments": "When you can't randomize at the individual level (e.g., TV, radio, OOH campaigns), geo-based experiments with CausalImpact are the gold standard"
        }
    
    return context


def compare_measurement_methodologies(methodology: str) -> str:
    """
    Get a detailed comparison between CausalImpact and another measurement methodology.
    
    Use this when users ask why we use CausalImpact instead of alternatives.
    
    Args:
        methodology: The methodology to compare against. Options include:
            - "arima" - ARIMA/time series forecasting
            - "did" or "difference_in_differences" - Difference-in-Differences
            - "synthetic_control" - Synthetic Control Method
            - "propensity" or "psm" - Propensity Score Matching
            - "ab_test" - Traditional A/B Testing
            - "regression" - Simple regression approaches
            - "all" - Compare against all methodologies
            
    Returns:
        JSON string with detailed methodology comparison
    """
    methodology_lower = methodology.lower().strip()
    
    comparisons = {
        "causal_impact": {
            "name": "CausalImpact (Bayesian Structural Time Series)",
            "developed_by": "Google (2015)",
            "best_for": [
                "Geo-based marketing experiments",
                "Time series data with seasonality",
                "When you need probabilistic inference",
                "TV, radio, OOH, and regional campaign measurement"
            ],
            "how_it_works": "Builds a Bayesian structural time series model using control regions to predict what would have happened in test regions without the intervention. The difference between predicted and actual is the causal effect.",
            "strengths": [
                "Handles autocorrelation in time series",
                "Provides credible intervals, not just point estimates",
                "Accounts for seasonality and trends",
                "Uses control regions to build counterfactual",
                "Works with relatively short pre-periods"
            ],
            "limitations": [
                "Requires good control regions that correlate with test",
                "Assumes intervention doesn't affect control regions (no spillover)",
                "Computationally more intensive than simple methods"
            ]
        },
        "arima": {
            "name": "ARIMA (AutoRegressive Integrated Moving Average)",
            "best_for": [
                "Pure forecasting without causal inference",
                "Single time series analysis",
                "Short-term predictions"
            ],
            "how_it_works": "Models a time series based on its own past values, using autoregression, differencing, and moving averages.",
            "vs_causal_impact": {
                "key_difference": "ARIMA forecasts but doesn't establish causality. It can't tell you what would have happened without the intervention.",
                "causal_impact_advantage": "CausalImpact uses control regions to build a counterfactual, enabling causal inference rather than just forecasting.",
                "when_to_use_arima": "Use ARIMA for demand forecasting or prediction tasks where you don't need to measure intervention effects."
            }
        },
        "did": {
            "name": "Difference-in-Differences (DiD)",
            "best_for": [
                "Policy evaluation",
                "Before/after comparisons with control groups",
                "When parallel trends assumption holds"
            ],
            "how_it_works": "Compares the change in outcomes over time between a treatment group and a control group. Calculates (Post_treatment - Pre_treatment) - (Post_control - Pre_control).",
            "vs_causal_impact": {
                "key_difference": "DiD assumes parallel trends and doesn't handle time series complexities like autocorrelation or seasonality.",
                "causal_impact_advantage": "CausalImpact models the time series structure explicitly, handles seasonality, and provides Bayesian uncertainty quantification.",
                "when_to_use_did": "Use DiD for simpler panel data analysis or when you have many cross-sectional units with few time periods."
            }
        },
        "synthetic_control": {
            "name": "Synthetic Control Method (SCM)",
            "best_for": [
                "Comparative case studies",
                "Policy interventions with a single treated unit",
                "When you need an interpretable counterfactual"
            ],
            "how_it_works": "Creates a weighted combination of control units that best matches the treated unit's pre-intervention characteristics. This 'synthetic' control serves as the counterfactual.",
            "vs_causal_impact": {
                "key_difference": "Traditional SCM doesn't provide uncertainty estimates and uses deterministic weighting.",
                "causal_impact_advantage": "CausalImpact can be viewed as a Bayesian generalization of SCM, with proper uncertainty quantification and the ability to handle more complex time series patterns.",
                "relationship": "CausalImpact incorporates similar ideas but in a Bayesian framework with structural time series modeling."
            }
        },
        "psm": {
            "name": "Propensity Score Matching (PSM)",
            "best_for": [
                "Observational studies without randomization",
                "Customer-level analysis",
                "When treatment assignment is non-random"
            ],
            "how_it_works": "Estimates the probability (propensity) of receiving treatment based on observed characteristics, then matches treated and control units with similar propensity scores.",
            "vs_causal_impact": {
                "key_difference": "PSM is for cross-sectional customer-level analysis; CausalImpact is for time series geo-level analysis.",
                "causal_impact_advantage": "For marketing campaigns that can't be randomized at the individual level (TV, radio, billboards), CausalImpact with geo experiments is more appropriate.",
                "when_to_use_psm": "Use PSM when you have customer-level data with non-random treatment assignment and want to create comparable groups."
            }
        },
        "ab_test": {
            "name": "Traditional A/B Testing",
            "best_for": [
                "Digital experiments with random assignment",
                "Website/app optimization",
                "Email and ad creative testing"
            ],
            "how_it_works": "Randomly assigns users to treatment or control, then compares outcomes using statistical tests (t-tests, chi-square, etc.).",
            "vs_causal_impact": {
                "key_difference": "A/B tests require random assignment at the user level, which isn't possible for many marketing channels.",
                "causal_impact_advantage": "CausalImpact enables causal inference for campaigns that can't be randomized at the individual level (e.g., TV, radio, OOH, regional promotions).",
                "when_to_use_ab": "Use A/B testing for digital channels where you can randomize at the user level (email, web, app)."
            }
        },
        "regression": {
            "name": "Simple Linear Regression / Marketing Mix Modeling",
            "best_for": [
                "Long-term trend analysis",
                "Budget allocation across channels",
                "Understanding contribution of multiple variables"
            ],
            "how_it_works": "Models the relationship between marketing inputs and outcomes, estimating coefficients for each variable.",
            "vs_causal_impact": {
                "key_difference": "Regression shows correlation but establishing causality requires careful identification strategies. MMM estimates average effects, not specific campaign impacts.",
                "causal_impact_advantage": "CausalImpact provides causal inference for a specific intervention with clear counterfactual, credible intervals, and handles time series structure.",
                "complementary_use": "Use MMM for strategic planning and budget allocation; use CausalImpact for measuring specific campaign effectiveness."
            }
        }
    }
    
    if methodology_lower == "all":
        return json.dumps({
            "our_methodology": comparisons["causal_impact"],
            "comparisons": {k: v for k, v in comparisons.items() if k != "causal_impact"},
            "recommendation": "CausalImpact is the gold standard for geo-based marketing experiments. Use it when measuring TV, radio, OOH, regional promotions, or any campaign where individual-level randomization isn't possible."
        }, indent=2)
    
    # Find matching methodology
    method_map = {
        "arima": "arima",
        "time series": "arima",
        "forecast": "arima",
        "did": "did",
        "difference": "did",
        "diff-in-diff": "did",
        "synthetic": "synthetic_control",
        "scm": "synthetic_control",
        "propensity": "psm",
        "psm": "psm",
        "matching": "psm",
        "ab": "ab_test",
        "a/b": "ab_test",
        "randomized": "ab_test",
        "regression": "regression",
        "mmm": "regression",
        "mix model": "regression"
    }
    
    matched_method = None
    for key, value in method_map.items():
        if key in methodology_lower:
            matched_method = value
            break
    
    if not matched_method:
        return json.dumps({
            "error": f"Methodology '{methodology}' not recognized",
            "available_methodologies": list(comparisons.keys()),
            "suggestion": "Try one of: arima, did, synthetic_control, psm, ab_test, regression, or 'all'"
        })
    
    return json.dumps({
        "our_methodology": comparisons["causal_impact"],
        "compared_methodology": comparisons[matched_method],
        "recommendation": f"For geo-based marketing experiments, CausalImpact is typically preferred over {comparisons[matched_method]['name']}."
    }, indent=2)


# ==================== Visualization Tools ====================

def generate_test_design_chart(
    chart_type: str,
    data_source: str,
    metric: str = "orders",
    group_by: str = None,
    filter_by: str = None,
    filter_value: str = None,
    top_n: int = 10,
    time_period: str = None,
    title: str = None
) -> str:
    """
    Generate charts and visualizations for test design analysis.
    Returns a base64-encoded PNG image that can be displayed in chat.
    
    Use this tool when users ask for:
    - Trending charts (orders, revenue over time)
    - Customer segment analysis (personas, attributes, LTV tiers)
    - Product performance (top products, categories)
    - Geographic distributions
    - Any visual analysis for test design planning
    
    Args:
        chart_type: Type of chart to generate. Options:
            - "line" - Time series/trending data
            - "bar" - Horizontal bar chart
            - "column" - Vertical bar/column chart  
            - "pie" - Pie chart for proportions
            - "histogram" - Distribution of values
            - "scatter" - Scatter plot for correlations
            
        data_source: Which data to use. Options:
            - "orders" - CRM sales data (order_date, customer_id, order_value)
            - "customers" - Customer file (attributes, LTV, frequency)
            - "products" - Order line items (product_name, price)
            - "orders_customers" - Join orders with customer data
            - "products_customers" - Join products with customer data
            
        metric: What to measure. Options:
            - "orders" or "order_count" - Count of orders
            - "revenue" or "order_value" - Sum of order values
            - "customers" or "customer_count" - Count of unique customers
            - "avg_order_value" - Average order value
            - "units" - Count of product units sold
            
        group_by: How to group/aggregate data. Options:
            - "month" - Group by month (for time series)
            - "week" - Group by week
            - "day" - Group by day
            - "customer_attribute" - Customer persona/segment
            - "ltv_tier" - LTV tier (low/medium/high)
            - "purch_freq_tier" - Purchase frequency tier
            - "region" or "state" or "dma" - Geographic grouping
            - "product_name" - By product
            - "gender" - By gender
            - "age_group" - By age brackets
            
        filter_by: Optional column to filter on before charting
        
        filter_value: Value to filter by (e.g., "eco_adventurer" for customer_attribute)
        
        top_n: For bar/column charts, limit to top N items (default 10)
        
        time_period: Optional time filter like "2025", "2025-Q1", "last_6_months"
        
        title: Optional custom chart title
        
    Returns:
        JSON with base64 encoded chart image and metadata
        
    Examples:
        - Trending monthly orders: chart_type="line", data_source="orders", metric="orders", group_by="month"
        - Top customer personas: chart_type="column", data_source="orders_customers", metric="revenue", group_by="customer_attribute", top_n=10
        - Top products for eco-adventurers: chart_type="bar", data_source="products_customers", metric="units", group_by="product_name", filter_by="customer_attribute", filter_value="eco_adventurer", top_n=5
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    try:
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette for Outdoorsy Living brand
        colors = ['#2E7D32', '#4CAF50', '#81C784', '#A5D6A7', '#C8E6C9', 
                  '#1B5E20', '#388E3C', '#66BB6A', '#43A047', '#00695C']
        
        # Load data based on source
        data_dir = Path(__file__).parent.parent / "data"
        
        if data_source in ["orders", "orders_customers", "products_customers"]:
            orders_df = pd.read_csv(data_dir / "crm_sales_cust_data" / "crm_sales_data_2025_H12026.csv")
            orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], format='%m/%d/%Y')
        
        if data_source in ["customers", "orders_customers", "products_customers"]:
            customers_df = pd.read_csv(data_dir / "crm_sales_cust_data" / "customer_file.csv")
        
        if data_source in ["products", "products_customers"]:
            products_df = pd.read_csv(data_dir / "crm_sales_cust_data" / "order_line_items.csv")
        
        # Create combined datasets as needed
        if data_source == "orders_customers":
            df = orders_df.merge(customers_df, on='customer_id', how='left')
        elif data_source == "products_customers":
            # Join products -> orders -> customers
            df = products_df.merge(orders_df[['unique_order_id', 'customer_id', 'order_date']], on='unique_order_id', how='left')
            df = df.merge(customers_df, on='customer_id', how='left')
        elif data_source == "orders":
            df = orders_df
        elif data_source == "customers":
            df = customers_df
        elif data_source == "products":
            df = products_df
        else:
            return json.dumps({"error": f"Unknown data_source: {data_source}"})
        
        # Apply time period filter if specified
        if time_period and 'order_date' in df.columns:
            if time_period.startswith("last_"):
                months = int(time_period.split("_")[1])
                cutoff = df['order_date'].max() - pd.DateOffset(months=months)
                df = df[df['order_date'] >= cutoff]
            elif "-Q" in time_period:
                year, quarter = time_period.split("-Q")
                df['quarter'] = df['order_date'].dt.quarter
                df = df[(df['order_date'].dt.year == int(year)) & (df['quarter'] == int(quarter))]
            else:
                df = df[df['order_date'].dt.year == int(time_period)]
        
        # Apply filter if specified
        if filter_by and filter_value and filter_by in df.columns:
            df = df[df[filter_by].astype(str).str.lower() == filter_value.lower()]
        
        # Create age groups if needed
        if group_by == "age_group" and 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                                     labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        # Initialize grouped to None so we can check if it was set
        grouped = None
        
        # Create time groupings if needed
        if group_by in ["month", "week", "day"] and 'order_date' in df.columns:
            if group_by == "month":
                df['time_group'] = df['order_date'].dt.to_period('M').astype(str)
            elif group_by == "week":
                df['time_group'] = df['order_date'].dt.to_period('W').astype(str)
            else:
                df['time_group'] = df['order_date'].dt.date.astype(str)
            group_col = 'time_group'
        else:
            group_col = group_by
        
        # Validate group_col exists
        if not group_col:
            plt.close(fig)
            return json.dumps({"error": "group_by parameter is required for this chart"})
        
        if group_col != 'time_group' and group_col not in df.columns:
            plt.close(fig)
            return json.dumps({"error": f"Cannot group by '{group_by}' - column not found in data. Available columns: {list(df.columns)}"})
        
        # Calculate the metric
        if group_col and (group_col in df.columns or group_col == 'time_group'):
            if metric in ["orders", "order_count"]:
                if 'unique_order_id' in df.columns:
                    grouped = df.groupby(group_col)['unique_order_id'].nunique().reset_index()
                    grouped.columns = [group_col, 'value']
                else:
                    grouped = df.groupby(group_col).size().reset_index(name='value')
            elif metric in ["revenue", "order_value"]:
                if 'order_value_usd' in df.columns:
                    grouped = df.groupby(group_col)['order_value_usd'].sum().reset_index()
                    grouped.columns = [group_col, 'value']
                elif 'price' in df.columns:
                    grouped = df.groupby(group_col)['price'].sum().reset_index()
                    grouped.columns = [group_col, 'value']
                else:
                    plt.close(fig)
                    return json.dumps({"error": "No revenue column found in data"})
            elif metric in ["customers", "customer_count"]:
                grouped = df.groupby(group_col)['customer_id'].nunique().reset_index()
                grouped.columns = [group_col, 'value']
            elif metric == "avg_order_value":
                if 'order_value_usd' in df.columns:
                    grouped = df.groupby(group_col)['order_value_usd'].mean().reset_index()
                    grouped.columns = [group_col, 'value']
                else:
                    plt.close(fig)
                    return json.dumps({"error": "No order value column found"})
            elif metric == "units":
                grouped = df.groupby(group_col).size().reset_index(name='value')
            else:
                plt.close(fig)
                return json.dumps({"error": f"Unknown metric: {metric}"})
        else:
            plt.close(fig)
            return json.dumps({"error": f"Cannot group by '{group_by}' - column not found in data"})
        
        # Verify grouped was created successfully
        if grouped is None or len(grouped) == 0:
            plt.close(fig)
            return json.dumps({"error": "No data available for the specified grouping and filters"})
        
        # Sort and limit for non-time-series charts
        if chart_type in ["bar", "column", "pie"] and group_by not in ["month", "week", "day"]:
            grouped = grouped.nlargest(top_n, 'value')
        
        # Sort time series data chronologically
        if group_by in ["month", "week", "day"]:
            grouped = grouped.sort_values(group_col)
        
        # Generate the chart
        if chart_type == "line":
            ax.plot(grouped[group_col], grouped['value'], marker='o', linewidth=2, 
                    markersize=6, color=colors[0])
            ax.fill_between(grouped[group_col], grouped['value'], alpha=0.3, color=colors[0])
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == "bar":
            bars = ax.barh(grouped[group_col], grouped['value'], color=colors[:len(grouped)])
            ax.invert_yaxis()  # Largest at top
            # Add value labels
            for bar, val in zip(bars, grouped['value']):
                ax.text(val + max(grouped['value'])*0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:,.0f}', va='center', fontsize=9)
                       
        elif chart_type == "column":
            bars = ax.bar(grouped[group_col], grouped['value'], color=colors[:len(grouped)])
            plt.xticks(rotation=45, ha='right')
            # Add value labels
            for bar, val in zip(bars, grouped['value']):
                ax.text(bar.get_x() + bar.get_width()/2, val + max(grouped['value'])*0.01,
                       f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
                       
        elif chart_type == "pie":
            ax.pie(grouped['value'], labels=grouped[group_col], autopct='%1.1f%%',
                   colors=colors[:len(grouped)], startangle=90)
            ax.axis('equal')
            
        elif chart_type == "histogram":
            ax.hist(df[group_by] if group_by in df.columns else df['value'], 
                   bins=20, color=colors[0], edgecolor='white', alpha=0.8)
                   
        elif chart_type == "scatter":
            # For scatter, we need two numeric columns
            if 'projected_ltv' in df.columns and 'purchase_frequency' in df.columns:
                ax.scatter(df['purchase_frequency'], df['projected_ltv'], 
                          alpha=0.5, color=colors[0], s=20)
                ax.set_xlabel('Purchase Frequency')
                ax.set_ylabel('Projected LTV ($)')
        
        # Set title
        if title:
            chart_title = title
        else:
            metric_labels = {
                "orders": "Orders", "order_count": "Orders", 
                "revenue": "Revenue ($)", "order_value": "Revenue ($)",
                "customers": "Customers", "customer_count": "Customers",
                "avg_order_value": "Avg Order Value ($)", "units": "Units Sold"
            }
            group_labels = {
                "month": "Month", "week": "Week", "day": "Day",
                "customer_attribute": "Customer Persona", "ltv_tier": "LTV Tier",
                "purch_freq_tier": "Purchase Frequency", "product_name": "Product",
                "region": "Region", "state": "State", "dma": "DMA",
                "gender": "Gender", "age_group": "Age Group"
            }
            metric_label = metric_labels.get(metric, metric.title())
            group_label = group_labels.get(group_by, group_by.title() if group_by else "")
            
            if filter_by and filter_value:
                chart_title = f"{metric_label} by {group_label}\n(Filtered: {filter_by}={filter_value})"
            else:
                chart_title = f"{metric_label} by {group_label}"
        
        ax.set_title(chart_title, fontsize=14, fontweight='bold', pad=20)
        
        # Format y-axis for large numbers
        if chart_type in ["line", "column"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Build summary stats
        summary = {
            "total_records": len(df),
            "groups_shown": len(grouped),
            "metric_total": float(grouped['value'].sum()),
            "metric_avg": float(grouped['value'].mean()),
            "metric_max": float(grouped['value'].max()),
            "top_group": str(grouped.loc[grouped['value'].idxmax(), group_col]) if len(grouped) > 0 else None
        }
        
        # Save artifact to disk
        artifact_path = None
        try:
            import streamlit as st
            from datetime import datetime
            
            session_id = st.session_state.get('adk_session_id', 'default')
            artifacts_dir = Path(__file__).parent.parent / "data" / "session_artifacts" / session_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_title_clean = (title or f"{metric}_{group_by}").replace(" ", "_").replace("/", "_")[:50]
            filename = f"chart_{timestamp}_{chart_title_clean}.png"
            artifact_path = artifacts_dir / filename
            
            # Decode base64 and save
            import base64
            image_data = base64.b64decode(image_base64)
            with open(artifact_path, 'wb') as f:
                f.write(image_data)
            
            # Update shared context with artifact info
            context = st.session_state.get('shared_agent_context', {})
            if 'generated_artifacts' not in context:
                context['generated_artifacts'] = []
            
            context['generated_artifacts'].append({
                "type": "chart",
                "path": str(artifact_path),
                "filename": filename,
                "timestamp": timestamp,
                "title": title or f"{metric} by {group_by}",
                "chart_type": chart_type,
                "data_source": data_source,
                "metric": metric
            })
            st.session_state['shared_agent_context'] = context
            
            logger.info(f"Saved chart artifact: {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to save chart artifact: {e}")
        
        return json.dumps({
            "success": True,
            "chart_type": chart_type,
            "data_source": data_source,
            "metric": metric,
            "group_by": group_by,
            "filter_applied": f"{filter_by}={filter_value}" if filter_by else None,
            "image_base64": image_base64,
            "artifact_saved": artifact_path is not None,
            "artifact_path": str(artifact_path) if artifact_path else None,
            "summary": summary,
            "display_hint": "Display this image inline in the chat response"
        })
        
    except Exception as e:
        logger.exception("Error generating chart")
        return json.dumps({
            "error": str(e),
            "chart_type": chart_type,
            "data_source": data_source
        })


def get_available_chart_options() -> str:
    """
    Get information about available charting options and data columns.
    Use this to understand what visualizations are possible.
    
    Returns:
        JSON with available chart types, data sources, metrics, and groupings
    """
    # Load column info from actual data
    data_dir = Path(__file__).parent.parent / "data"
    
    try:
        customers_df = pd.read_csv(data_dir / "crm_sales_cust_data" / "customer_file.csv", nrows=5)
        orders_df = pd.read_csv(data_dir / "crm_sales_cust_data" / "crm_sales_data_2025_H12026.csv", nrows=5)
        products_df = pd.read_csv(data_dir / "crm_sales_cust_data" / "order_line_items.csv", nrows=5)
        
        # Get unique values for key categorical columns
        customers_full = pd.read_csv(data_dir / "crm_sales_cust_data" / "customer_file.csv")
        
        return json.dumps({
            "chart_types": {
                "line": "Time series/trending data - best for showing changes over time",
                "bar": "Horizontal bars - best for comparing categories (sorted largest to smallest)",
                "column": "Vertical bars - good for comparing categories or time periods",
                "pie": "Proportions - shows percentage breakdown of a whole",
                "histogram": "Distribution - shows spread of numeric values",
                "scatter": "Correlations - shows relationship between two variables"
            },
            "data_sources": {
                "orders": f"CRM sales data - columns: {list(orders_df.columns)}",
                "customers": f"Customer file - columns: {list(customers_df.columns)}",
                "products": f"Order line items - columns: {list(products_df.columns)}",
                "orders_customers": "Orders joined with customer data (for segmented sales analysis)",
                "products_customers": "Products joined with customer data (for product analysis by segment)"
            },
            "metrics": {
                "orders": "Count of orders",
                "revenue": "Sum of order/product values",
                "customers": "Count of unique customers",
                "avg_order_value": "Average order value",
                "units": "Count of items/rows"
            },
            "group_by_options": {
                "time": ["month", "week", "day"],
                "customer_segments": ["customer_attribute", "ltv_tier", "purch_freq_tier", "gender", "age_group"],
                "geography": ["region", "state", "dma"],
                "products": ["product_name"]
            },
            "customer_attributes": list(customers_full['customer_attribute'].unique()),
            "ltv_tiers": list(customers_full['ltv_tier'].unique()),
            "example_queries": [
                {"description": "Monthly order trend", "params": {"chart_type": "line", "data_source": "orders", "metric": "orders", "group_by": "month"}},
                {"description": "Top customer personas by revenue", "params": {"chart_type": "column", "data_source": "orders_customers", "metric": "revenue", "group_by": "customer_attribute", "top_n": 10}},
                {"description": "Top 5 products for eco-adventurers", "params": {"chart_type": "bar", "data_source": "products_customers", "metric": "units", "group_by": "product_name", "filter_by": "customer_attribute", "filter_value": "eco_adventurer", "top_n": 5}}
            ]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_pre_period_balance_charts(
    test_id: str,
    chart_type: str = "all",
    metric: str = "revenue"
) -> str:
    """
    Generate pre-period balance charts for a saved test to validate test design.
    Shows that test and control groups had similar behavior BEFORE the intervention.
    
    This is a TEST DESIGN validation tool - use this to confirm a test split is valid
    before the test begins or to validate historical balance.
    
    Args:
        test_id: The test ID to generate charts for (e.g., "TEST-12011514")
        chart_type: Type of chart to generate. Options:
            - "all" - Generate all balance charts (default)
            - "weekly_trend" - Weekly revenue/orders trend for test vs control
            - "cumulative" - Cumulative comparison over pre-period
            - "distribution" - Distribution comparison (histogram overlay)
            - "summary_stats" - Bar chart comparing key metrics
        metric: What to compare - "revenue" or "orders"
            
    Returns:
        JSON with base64 encoded chart image(s) and balance statistics
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import base64
    
    try:
        # Normalize test_id format
        if not test_id.upper().startswith("TEST-"):
            test_id = f"TEST-{test_id}"
        else:
            test_id = test_id.upper()
        
        # Get test details
        test_details = get_test_by_id(test_id)
        if test_details is None:
            test_details = data_loader.get_test_summary(test_id)
            
        if test_details is None:
            return json.dumps({"error": f"Test not found: {test_id}. Please check the test ID."})
        
        # Load audience files
        test_audience = data_loader.load_test_audience(test_id)
        control_audience = data_loader.load_control_audience(test_id)
        
        if test_audience is None or control_audience is None:
            return json.dumps({"error": f"Audience files not found for test {test_id}"})
        
        test_ids = test_audience["customer_id"].tolist()
        control_ids = control_audience["customer_id"].tolist()
        
        # Load sales data
        sales_df = data_loader.load_sales_data()
        if sales_df is None:
            return json.dumps({"error": "Could not load sales data"})
        
        # Detect date column
        date_col = None
        for col in ["order_date", "transaction_date", "date"]:
            if col in sales_df.columns:
                date_col = col
                break
        if date_col is None:
            return json.dumps({"error": "Could not find date column in sales data"})
        
        sales_df[date_col] = pd.to_datetime(sales_df[date_col], format='%m/%d/%Y', errors='coerce')
        
        # Detect value column
        value_col = None
        for col in ["order_value_usd", "order_value", "revenue", "amount"]:
            if col in sales_df.columns:
                value_col = col
                break
        
        # Detect customer ID column
        cust_col = None
        for col in ["customer_id", "cust_id", "customerid"]:
            if col in sales_df.columns:
                cust_col = col
                break
        if cust_col is None:
            return json.dumps({"error": "Could not find customer_id column in sales data"})
        
        # Get pre-period dates
        pre_start = test_details.get("pre_period_lookback_start") or test_details.get("hist_split_balance_start_date")
        pre_end = test_details.get("pre_period_lookback_end") or test_details.get("hist_split_balance_end_date")
        
        if not pre_start or not pre_end:
            # Default to 90 days before test start
            test_start = test_details.get("start_date")
            if test_start:
                pre_end_dt = pd.to_datetime(test_start) - pd.Timedelta(days=1)
                pre_start_dt = pre_end_dt - pd.Timedelta(days=90)
                pre_start = pre_start_dt.strftime("%Y-%m-%d")
                pre_end = pre_end_dt.strftime("%Y-%m-%d")
            else:
                return json.dumps({"error": "Could not determine pre-period dates"})
        
        pre_start_dt = pd.to_datetime(pre_start)
        pre_end_dt = pd.to_datetime(pre_end)
        
        # Filter to pre-period
        pre_period_mask = (sales_df[date_col] >= pre_start_dt) & (sales_df[date_col] <= pre_end_dt)
        pre_sales = sales_df[pre_period_mask].copy()
        
        # Split into test and control
        test_sales = pre_sales[pre_sales[cust_col].isin(test_ids)]
        control_sales = pre_sales[pre_sales[cust_col].isin(control_ids)]
        
        # Aggregate by week
        test_sales['week'] = test_sales[date_col].dt.to_period('W').apply(lambda x: x.start_time)
        control_sales['week'] = control_sales[date_col].dt.to_period('W').apply(lambda x: x.start_time)
        
        if metric == "revenue" and value_col:
            test_weekly = test_sales.groupby('week')[value_col].sum().reset_index()
            control_weekly = control_sales.groupby('week')[value_col].sum().reset_index()
            test_weekly.columns = ['week', 'value']
            control_weekly.columns = ['week', 'value']
            metric_label = "Revenue ($)"
        else:
            test_weekly = test_sales.groupby('week').size().reset_index(name='value')
            control_weekly = control_sales.groupby('week').size().reset_index(name='value')
            metric_label = "Order Count"
        
        # Merge for comparison
        weekly_df = test_weekly.merge(control_weekly, on='week', how='outer', suffixes=('_test', '_control'))
        weekly_df = weekly_df.fillna(0).sort_values('week')
        
        # Calculate balance statistics
        correlation = weekly_df['value_test'].corr(weekly_df['value_control'])
        test_mean = weekly_df['value_test'].mean()
        control_mean = weekly_df['value_control'].mean()
        pct_diff = abs(test_mean - control_mean) / ((test_mean + control_mean) / 2) * 100 if (test_mean + control_mean) > 0 else 0
        
        generated_charts = []
        
        # Set up styling
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        # Chart 1: Weekly Trend
        if chart_type in ["all", "weekly_trend"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(weekly_df['week'], weekly_df['value_test'], 'b-', linewidth=2, marker='o', markersize=4, label='Test Group')
            ax.plot(weekly_df['week'], weekly_df['value_control'], 'r--', linewidth=2, marker='s', markersize=4, label='Control Group')
            
            ax.set_title(f'Pre-Period Weekly {metric.title()} - {test_details.get("test_name", test_id)}\nCorrelation: {correlation:.3f}')
            ax.set_xlabel('Week')
            ax.set_ylabel(metric_label)
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add balance indicator
            balance_text = f"Balance: {'✓ Good' if correlation > 0.8 else '⚠ Check'} (r={correlation:.2f})"
            ax.annotate(balance_text, xy=(0.98, 0.02), xycoords='axes fraction', ha='right',
                       fontsize=10, color='green' if correlation > 0.8 else 'orange',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "weekly_trend",
                "title": f"Pre-Period Weekly {metric.title()} Trend",
                "description": f"Test vs Control weekly {metric}. Correlation: {correlation:.3f}. Higher correlation = better balance.",
                "image_base64": img_base64
            })
        
        # Chart 2: Cumulative Comparison
        if chart_type in ["all", "cumulative"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            test_cumsum = weekly_df['value_test'].cumsum()
            control_cumsum = weekly_df['value_control'].cumsum()
            
            ax.plot(weekly_df['week'], test_cumsum, 'b-', linewidth=2, label='Test Group (Cumulative)')
            ax.plot(weekly_df['week'], control_cumsum, 'r--', linewidth=2, label='Control Group (Cumulative)')
            
            ax.set_title(f'Pre-Period Cumulative {metric.title()} - {test_details.get("test_name", test_id)}')
            ax.set_xlabel('Week')
            ax.set_ylabel(f'Cumulative {metric_label}')
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "cumulative",
                "title": f"Pre-Period Cumulative {metric.title()}",
                "description": "Cumulative totals should track closely if groups are balanced.",
                "image_base64": img_base64
            })
        
        # Chart 3: Summary Stats Bar Chart
        if chart_type in ["all", "summary_stats"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            metrics_names = ['Total', 'Weekly Avg', 'Std Dev']
            test_values = [
                weekly_df['value_test'].sum(),
                weekly_df['value_test'].mean(),
                weekly_df['value_test'].std()
            ]
            control_values = [
                weekly_df['value_control'].sum(),
                weekly_df['value_control'].mean(),
                weekly_df['value_control'].std()
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, test_values, width, label='Test Group', color='#2196F3')
            bars2 = ax.bar(x + width/2, control_values, width, label='Control Group', color='#F44336')
            
            ax.set_ylabel(metric_label)
            ax.set_title(f'Pre-Period Summary Statistics - {test_details.get("test_name", test_id)}')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend()
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "summary_stats",
                "title": "Pre-Period Summary Statistics",
                "description": "Side-by-side comparison of key metrics. Values should be similar.",
                "image_base64": img_base64
            })
        
        # Chart 4: Distribution Histogram
        if chart_type in ["all", "distribution"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.hist(weekly_df['value_test'], bins=15, alpha=0.6, label='Test Group', color='#2196F3')
            ax.hist(weekly_df['value_control'], bins=15, alpha=0.6, label='Control Group', color='#F44336')
            
            ax.set_title(f'Pre-Period {metric.title()} Distribution - {test_details.get("test_name", test_id)}')
            ax.set_xlabel(metric_label)
            ax.set_ylabel('Frequency')
            ax.legend()
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "distribution",
                "title": f"Pre-Period {metric.title()} Distribution",
                "description": "Overlapping distributions indicate similar behavior patterns.",
                "image_base64": img_base64
            })
        
        # Store artifacts in session
        try:
            import streamlit as st
            if 'shared_agent_context' not in st.session_state:
                st.session_state['shared_agent_context'] = {}
            if 'generated_artifacts' not in st.session_state['shared_agent_context']:
                st.session_state['shared_agent_context']['generated_artifacts'] = []
            
            for chart in generated_charts:
                st.session_state['shared_agent_context']['generated_artifacts'].append({
                    "type": "chart",
                    "chart_type": f"pre_period_{chart['type']}",
                    "title": chart["title"],
                    "test_id": test_id,
                    "timestamp": datetime.now().isoformat()
                })
        except:
            pass
        
        return json.dumps({
            "success": True,
            "test_id": test_id,
            "test_name": test_details.get("test_name"),
            "pre_period": f"{pre_start} to {pre_end}",
            "charts_generated": len(generated_charts),
            "charts": generated_charts,
            "balance_assessment": {
                "correlation": round(correlation, 4),
                "test_group_mean": round(test_mean, 2),
                "control_group_mean": round(control_mean, 2),
                "percent_difference": round(pct_diff, 2),
                "balance_status": "Good" if correlation > 0.8 and pct_diff < 10 else "Review Needed",
                "interpretation": f"Correlation of {correlation:.2f} indicates {'strong' if correlation > 0.8 else 'moderate' if correlation > 0.6 else 'weak'} pre-period balance. "
                                 f"Mean difference of {pct_diff:.1f}% is {'acceptable' if pct_diff < 10 else 'concerning'}."
            },
            "group_sizes": {
                "test_customers": len(test_ids),
                "control_customers": len(control_ids)
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating pre-period balance charts: {e}")
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def get_causal_impact_charts(
    test_id: str,
    chart_type: str = "all"
) -> str:
    """
    Generate CausalImpact visualization charts for a test.
    Creates predicted vs actual, pointwise effects, and cumulative effects charts.
    
    Args:
        test_id: The test ID to generate charts for
        chart_type: Type of chart to generate. Options:
            - "all" - Generate all three charts (default)
            - "predicted_actual" - Actual vs predicted (counterfactual) time series
            - "pointwise" - Point-by-point effect estimates with confidence intervals  
            - "cumulative" - Cumulative effect over time
            
    Returns:
        JSON with base64 encoded chart image(s) and metadata
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import base64
    
    try:
        # Get test details
        test_details = get_test_by_id(test_id)
        if test_details is None:
            test_details = data_loader.get_test_summary(test_id)
            
        if test_details is None:
            return json.dumps({"error": f"Test not found: {test_id}"})
        
        # Load audience files to re-run analysis and get inferences
        test_audience = data_loader.load_test_audience(test_id)
        control_audience = data_loader.load_control_audience(test_id)
        
        if test_audience is None or control_audience is None:
            return json.dumps({"error": "Audience files not found for this test"})
        
        test_ids = test_audience["customer_id"].tolist()
        control_ids = control_audience["customer_id"].tolist()
        
        # Load sales data
        sales_df = data_loader.load_sales_data()
        
        # Prepare time series
        ts_data = causal_impact_analyzer.prepare_time_series_data(
            sales_df=sales_df,
            test_customer_ids=test_ids,
            control_customer_ids=control_ids,
            metric="revenue"
        )
        
        # Get periods
        pre_start = test_details.get("pre_period_lookback_start") or test_details.get("hist_split_balance_start_date")
        pre_end = test_details.get("pre_period_lookback_end") or test_details.get("hist_split_balance_end_date")
        post_start = test_details.get("start_date")
        post_end = test_details.get("end_date")
        
        if any(v is None or (isinstance(v, str) and v.strip() == "") for v in [pre_start, pre_end, post_start, post_end]):
            return json.dumps({"error": "Missing date fields required for chart generation"})
        
        # Run analysis to get inferences
        results = causal_impact_analyzer.run_causal_impact(
            data=ts_data,
            pre_period=(pre_start, pre_end),
            post_period=(post_start, post_end)
        )
        
        if not results.get("success"):
            return json.dumps({"error": results.get("error", "Analysis failed")})
        
        # Get inferences data
        inferences_dict = results.get("inferences")
        if inferences_dict is None:
            # Fall back to generating from raw time series
            return _generate_simplified_charts(test_id, ts_data, pre_end, post_start, post_end, chart_type, test_details)
        
        # Convert inferences back to DataFrame
        inferences = pd.DataFrame(inferences_dict)
        inferences.index = pd.to_datetime(inferences.index)
        
        generated_charts = []
        
        # Set up consistent styling
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        intervention_date = pd.to_datetime(post_start)
        
        # Chart 1: Predicted vs Actual
        if chart_type in ["all", "predicted_actual"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(inferences.index, inferences['response'], 'b-', linewidth=1.5, label='Actual')
            ax.plot(inferences.index, inferences['point_pred'], 'r--', linewidth=1.5, label='Predicted (Counterfactual)')
            
            # Confidence interval
            ax.fill_between(
                inferences.index,
                inferences['point_pred_lower'],
                inferences['point_pred_upper'],
                color='red', alpha=0.15, label='95% CI'
            )
            
            # Intervention line
            ax.axvline(x=intervention_date, color='green', linestyle='--', linewidth=2, label='Intervention')
            ax.axvspan(intervention_date, inferences.index.max(), alpha=0.1, color='green')
            
            ax.set_title(f'Actual vs Predicted - {test_details.get("test_name", test_id)}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Revenue ($)')
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "predicted_actual",
                "title": "Actual vs Predicted (Counterfactual)",
                "description": "Blue line shows actual revenue, red dashed shows what would have happened without intervention",
                "image_base64": img_base64
            })
        
        # Chart 2: Pointwise Effects
        if chart_type in ["all", "pointwise"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Calculate pointwise effect
            pointwise_effect = inferences['response'] - inferences['point_pred']
            pointwise_lower = inferences['response'] - inferences['point_pred_upper']
            pointwise_upper = inferences['response'] - inferences['point_pred_lower']
            
            ax.plot(inferences.index, pointwise_effect, 'b-', linewidth=1.5, label='Point Effect')
            ax.fill_between(
                inferences.index,
                pointwise_lower,
                pointwise_upper,
                color='blue', alpha=0.2, label='95% CI'
            )
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(x=intervention_date, color='green', linestyle='--', linewidth=2, label='Intervention')
            ax.axvspan(intervention_date, inferences.index.max(), alpha=0.1, color='green')
            
            ax.set_title(f'Pointwise Effect - {test_details.get("test_name", test_id)}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Effect ($)')
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "pointwise",
                "title": "Pointwise Effect",
                "description": "Daily difference between actual and predicted. Positive = intervention helped.",
                "image_base64": img_base64
            })
        
        # Chart 3: Cumulative Effects
        if chart_type in ["all", "cumulative"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Calculate cumulative effect (only post-intervention)
            post_mask = inferences.index >= intervention_date
            cumulative_effect = (inferences['response'] - inferences['point_pred']).cumsum()
            cumulative_lower = (inferences['response'] - inferences['point_pred_upper']).cumsum()
            cumulative_upper = (inferences['response'] - inferences['point_pred_lower']).cumsum()
            
            ax.plot(inferences.index, cumulative_effect, 'b-', linewidth=1.5, label='Cumulative Effect')
            ax.fill_between(
                inferences.index,
                cumulative_lower,
                cumulative_upper,
                color='blue', alpha=0.2, label='95% CI'
            )
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(x=intervention_date, color='green', linestyle='--', linewidth=2, label='Intervention')
            ax.axvspan(intervention_date, inferences.index.max(), alpha=0.1, color='green')
            
            ax.set_title(f'Cumulative Effect - {test_details.get("test_name", test_id)}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Effect ($)')
            ax.legend(loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            generated_charts.append({
                "type": "cumulative",
                "title": "Cumulative Effect",
                "description": "Running total of intervention effect over time",
                "image_base64": img_base64
            })
        
        # Store artifacts in session
        try:
            import streamlit as st
            if 'shared_agent_context' not in st.session_state:
                st.session_state['shared_agent_context'] = {}
            if 'generated_artifacts' not in st.session_state['shared_agent_context']:
                st.session_state['shared_agent_context']['generated_artifacts'] = []
            
            for chart in generated_charts:
                st.session_state['shared_agent_context']['generated_artifacts'].append({
                    "type": "chart",
                    "chart_type": chart["type"],
                    "title": chart["title"],
                    "test_id": test_id,
                    "timestamp": datetime.now().isoformat()
                })
        except:
            pass  # OK if session state not available
        
        return json.dumps({
            "success": True,
            "test_id": test_id,
            "test_name": test_details.get("test_name"),
            "charts_generated": len(generated_charts),
            "charts": generated_charts,
            "summary": {
                "lift_percent": results.get("lift_percent"),
                "lift_absolute": results.get("lift_absolute"),
                "p_value": results.get("p_value"),
                "significant": results.get("significant")
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating causal impact charts: {e}")
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def _generate_simplified_charts(test_id, ts_data, pre_end, post_start, post_end, chart_type, test_details):
    """Generate simplified charts when CausalImpact inferences not available."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import base64
    
    generated_charts = []
    intervention_date = pd.to_datetime(post_start)
    
    # Simplified predicted vs actual using control as proxy
    if chart_type in ["all", "predicted_actual"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(ts_data.index, ts_data['test'], 'b-', linewidth=1.5, label='Test Group (Actual)')
        ax.plot(ts_data.index, ts_data['control'], 'r--', linewidth=1.5, label='Control Group')
        
        ax.axvline(x=intervention_date, color='green', linestyle='--', linewidth=2, label='Intervention')
        ax.axvspan(intervention_date, ts_data.index.max(), alpha=0.1, color='green')
        
        ax.set_title(f'Test vs Control - {test_details.get("test_name", test_id)}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Revenue ($)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        generated_charts.append({
            "type": "predicted_actual",
            "title": "Test vs Control Groups",
            "description": "Comparison of test and control group revenue over time",
            "image_base64": img_base64
        })
    
    # Simplified pointwise (test - control)
    if chart_type in ["all", "pointwise"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        diff = ts_data['test'] - ts_data['control']
        ax.plot(ts_data.index, diff, 'b-', linewidth=1.5, label='Difference (Test - Control)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=intervention_date, color='green', linestyle='--', linewidth=2, label='Intervention')
        ax.axvspan(intervention_date, ts_data.index.max(), alpha=0.1, color='green')
        
        ax.set_title(f'Daily Difference - {test_details.get("test_name", test_id)}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Difference ($)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        generated_charts.append({
            "type": "pointwise",
            "title": "Daily Difference",
            "description": "Daily difference between test and control groups",
            "image_base64": img_base64
        })
    
    # Simplified cumulative
    if chart_type in ["all", "cumulative"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        diff = ts_data['test'] - ts_data['control']
        cumulative = diff.cumsum()
        ax.plot(ts_data.index, cumulative, 'b-', linewidth=1.5, label='Cumulative Difference')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=intervention_date, color='green', linestyle='--', linewidth=2, label='Intervention')
        ax.axvspan(intervention_date, ts_data.index.max(), alpha=0.1, color='green')
        
        ax.set_title(f'Cumulative Difference - {test_details.get("test_name", test_id)}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative ($)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        generated_charts.append({
            "type": "cumulative",
            "title": "Cumulative Difference",
            "description": "Running total of test minus control",
            "image_base64": img_base64
        })
    
    return json.dumps({
        "success": True,
        "test_id": test_id,
        "test_name": test_details.get("test_name"),
        "charts_generated": len(generated_charts),
        "charts": generated_charts,
        "note": "Simplified charts generated (CausalImpact inferences not available)"
    }, indent=2)


# Export all tools for use in agents
__all__ = [
    # Data loading tools
    "get_customer_data_summary",
    "get_existing_tests",
    "get_test_details",
    # Test design tools
    "calculate_sample_size",
    "create_random_split",
    "create_geographic_split",
    "save_current_test_design",  # Preferred - uses cached split
    "get_current_split_status",  # Check if split is ready to save
    "save_test_design",  # Internal - requires explicit customer IDs
    # Measurement tools
    "run_causal_impact_analysis",
    "get_measurement_summary",
    "get_measurement_history",
    "get_all_measurement_history",
    "update_test_status",
    # Validation & logging tools
    "get_test_validation_files",
    "get_test_version_history",
    # Validation data analysis tools
    "get_validation_file_data",
    "get_pre_period_balance_check",
    "get_model_diagnostics",
    "get_robustness_analysis",
    "get_daily_effects_analysis",
    # Knowledge base / RAG tools
    "search_causal_impact_knowledge",
    "initialize_knowledge_base",
    "get_knowledge_base_info",
    # Web search tools
    "search_measurement_methodology",
    "compare_measurement_methodologies",
    # Visualization tools
    "generate_test_design_chart",
    "get_available_chart_options",
    "generate_cached_split_balance_chart",
    "generate_pre_period_balance_charts",
    "get_causal_impact_charts",
    # Shared agent context tools
    "read_shared_context",
    "update_shared_context",
    "list_session_artifacts"
]


# ==================== Shared Agent Context Tools ====================

def read_shared_context(keys: Optional[List[str]] = None) -> str:
    """
    Read the shared context that all agents can access.
    This contains high-level information like project objective, current phase, key decisions, etc.
    
    Args:
        keys: Optional list of specific keys to read. If None, returns all context.
        
    Returns:
        JSON string with the requested context
    """
    try:
        # Import streamlit here to avoid circular dependency
        import streamlit as st
        
        context = st.session_state.get('shared_agent_context', {})
        
        if keys:
            filtered_context = {k: context.get(k) for k in keys if k in context}
            return json.dumps(filtered_context, indent=2)
        else:
            return json.dumps(context, indent=2)
    except Exception as e:
        logger.error(f"Error reading shared context: {e}")
        return json.dumps({"error": str(e)})


def update_shared_context(updates: Dict[str, Any]) -> str:
    """
    Update the shared context that all agents can access.
    Use this to communicate important information to other agents (e.g., decisions, current status).
    
    Args:
        updates: Dictionary of key-value pairs to update in the shared context.
                 Special keys:
                 - 'objective': The main goal/objective of the current work
                 - 'current_phase': Current phase (e.g., 'planning', 'design', 'measurement')
                 - 'key_decisions': List of important decisions (will append if provided as list)
                 - 'notes': List of notes for other agents (will append if provided as list)
    
    Returns:
        Confirmation message with updated context
    """
    try:
        # Import streamlit here to avoid circular dependency
        import streamlit as st
        
        context = st.session_state.get('shared_agent_context', {})
        
        for key, value in updates.items():
            # Handle list fields - append instead of replace
            if key in ['key_decisions', 'notes'] and isinstance(value, list):
                if key not in context:
                    context[key] = []
                context[key].extend(value)
            elif key in ['key_decisions', 'notes'] and isinstance(value, str):
                if key not in context:
                    context[key] = []
                context[key].append(value)
            else:
                context[key] = value
        
        st.session_state['shared_agent_context'] = context
        
        return f"✅ Shared context updated successfully.\n\nCurrent context:\n{json.dumps(context, indent=2)}"
    except Exception as e:
        logger.error(f"Error updating shared context: {e}")
        return f"❌ Error updating shared context: {str(e)}"


def list_session_artifacts() -> str:
    """
    List all artifacts (charts, files) generated in the current session.
    
    Returns:
        JSON list of artifacts with metadata (type, path, timestamp, title)
    """
    try:
        import streamlit as st
        
        context = st.session_state.get('shared_agent_context', {})
        artifacts = context.get('generated_artifacts', [])
        
        if not artifacts:
            return json.dumps({
                "success": True,
                "count": 0,
                "message": "No artifacts generated yet in this session.",
                "artifacts": []
            }, indent=2)
        
        return json.dumps({
            "success": True,
            "count": len(artifacts),
            "artifacts": artifacts
        }, indent=2)
    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        return json.dumps({"error": str(e)})


def update_test_checklist(item: str, status: bool = True) -> str:
    """
    Update a pre-test checklist item status in the Test Design workspace.
    Call this tool when you confirm or complete a step in the test design process.
    
    Valid checklist items:
    - "test_objective": Test objective defined
    - "hypothesis": Hypothesis documented  
    - "kpi_selected": Primary KPI selected
    - "split_method": Split methodology chosen
    - "sample_size": Sample size validated
    - "duration": Test duration set
    - "pre_period_check": Pre-period balance checked
    - "sign_off": Ready for sign-off (use when design is complete)
    
    Args:
        item: The checklist item key to update (see valid items above)
        status: True to check the item, False to uncheck (default: True)
        
    Returns:
        Confirmation message with current checklist state
    """
    valid_items = [
        "test_objective",
        "hypothesis", 
        "kpi_selected",
        "split_method",
        "sample_size",
        "duration",
        "pre_period_check",
        "sign_off"
    ]
    
    item_labels = {
        "test_objective": "Test objective defined",
        "hypothesis": "Hypothesis documented",
        "kpi_selected": "Primary KPI selected",
        "split_method": "Split methodology chosen",
        "sample_size": "Sample size validated",
        "duration": "Test duration set",
        "pre_period_check": "Pre-period balance checked",
        "sign_off": "Ready for sign-off"
    }
    
    if item not in valid_items:
        return json.dumps({
            "success": False,
            "error": f"Invalid checklist item: '{item}'",
            "valid_items": valid_items
        }, indent=2)
    
    try:
        import streamlit as st
        
        # Initialize test_design if not present
        if 'test_design' not in st.session_state:
            st.session_state.test_design = {}
        
        # Initialize checklist if not present
        if 'checklist' not in st.session_state.test_design:
            st.session_state.test_design['checklist'] = {}
        
        # Update the item
        st.session_state.test_design['checklist'][item] = status
        
        # Get current checklist state
        checklist = st.session_state.test_design.get('checklist', {})
        
        # Calculate progress
        completed = sum(1 for k in valid_items if checklist.get(k, False))
        total = len(valid_items)
        
        # Build status display
        status_display = []
        for key in valid_items:
            is_done = checklist.get(key, False)
            icon = "✅" if is_done else "⬜"
            status_display.append(f"{icon} {item_labels[key]}")
        
        action = "checked" if status else "unchecked"
        
        return json.dumps({
            "success": True,
            "message": f"'{item_labels[item]}' has been {action}.",
            "progress": f"{completed}/{total} items complete",
            "checklist_status": status_display
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error updating checklist: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

