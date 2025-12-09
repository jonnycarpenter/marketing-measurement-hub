"""
Logging Utilities for Marketing Measurement System

This module handles all logging logic including:
- Creating test validation folders
- Creating version history folders and files
- Generating validation CSV files at different stages
- Updating the master testing document
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEST_VALIDATION_DIR = os.path.join(DATA_DIR, "test_validation_files")
VERSION_HISTORY_DIR = os.path.join(DATA_DIR, "test_version_history")
MASTER_TESTING_DOC = os.path.join(DATA_DIR, "master_testing_doc.csv")


# =============================================================================
# FOLDER CREATION FUNCTIONS
# =============================================================================

def create_test_folders(test_id: str) -> Dict[str, str]:
    """
    Create the folder structure for a new test.
    
    Creates:
    - data/test_validation_files/{TEST_ID}_test_validation_files/
    - data/test_version_history/{TEST_ID}_version_history/
    
    Args:
        test_id: The unique test identifier (e.g., TEST-05011030)
        
    Returns:
        Dict with paths to created folders
    """
    # Create validation files folder
    validation_folder = os.path.join(TEST_VALIDATION_DIR, f"{test_id}_test_validation_files")
    os.makedirs(validation_folder, exist_ok=True)
    
    # Create version history folder
    version_folder = os.path.join(VERSION_HISTORY_DIR, f"{test_id}_version_history")
    os.makedirs(version_folder, exist_ok=True)
    
    logger.info(f"Created folders for test {test_id}")
    
    return {
        "validation_folder": validation_folder,
        "version_folder": version_folder
    }


def get_next_version_number(test_id: str) -> str:
    """
    Get the next version number for a test's version history.
    
    Args:
        test_id: The test identifier
        
    Returns:
        Version string like '001', '002', etc.
    """
    version_folder = os.path.join(VERSION_HISTORY_DIR, f"{test_id}_version_history")
    
    if not os.path.exists(version_folder):
        return "001"
    
    existing_files = [f for f in os.listdir(version_folder) if f.endswith('.csv')]
    if not existing_files:
        return "001"
    
    # Extract version numbers
    versions = []
    for f in existing_files:
        try:
            # Extract number from filename like TEST-001_version_history_001.csv
            version_str = f.split('_')[-1].replace('.csv', '')
            versions.append(int(version_str))
        except (ValueError, IndexError):
            continue
    
    if not versions:
        return "001"
    
    next_version = max(versions) + 1
    return f"{next_version:03d}"


# =============================================================================
# VALIDATION FILE SCHEMAS
# =============================================================================

# =============================================================================
# VALIDATION FILE SCHEMAS
# Per causal_impact_validation_spec.md - 8 core CSVs for measurement validation
# =============================================================================

VALIDATION_FILE_SCHEMAS = {
    # 1. pre_period_summary.csv - Balance check
    "pre_period_summary": [
        "metric_name", "treatment_avg", "control_avg", "pct_diff", 
        "t_stat", "p_value", "min_pre_period_date", "max_pre_period_date"
    ],
    
    # 2. pre_period_trends.csv - Parallel trends validation
    "pre_period_trends": [
        "performance_date", "treatment_value", "control_value", "diff_treatment_control"
    ],
    
    # 3. model_input_timeseries.csv - Full time series fed to model
    "model_input_timeseries": [
        "date", "treatment_series", "control_series", "synthetic_control_series"
    ],
    
    # 4. model_fit_stats.csv - Model diagnostic statistics
    "model_fit_stats": [
        "metric_name", "value", "description"
    ],
    
    # 5. daily_effects.csv - CRITICAL for charting (predicted vs actual, pointwise, cumulative)
    "daily_effects": [
        "date", "actual", "expected_counterfactual", "point_effect", 
        "lower_effect", "upper_effect"
    ],
    
    # 6. impact_summary.csv - Headline results
    "impact_summary": [
        "avg_lift", "cum_lift", "cred_int_lower", "cred_int_upper",
        "prob_lift_gt_0", "p_value_equivalent", "relative_lift_pct"
    ],
    
    # 7. model_residuals.csv - Residuals across periods for diagnostics
    "model_residuals": [
        "date", "residual_value", "is_pre_period", "is_post_period"
    ],
    
    # 8. robustness_checks.csv - Sensitivity analysis
    "robustness_checks": [
        "scenario_name", "avg_lift", "cred_int_lower", "cred_int_upper", "prob_lift_gt_0"
    ]
}

# Files created at test design finalization
DESIGN_PHASE_FILES = [
    "pre_period_summary",
    "pre_period_trends"
]

# Files created at measurement completion
# Per causal_impact_validation_spec.md - 8 core validation CSVs
MEASUREMENT_PHASE_FILES = [
    "pre_period_summary",      # Balance check (can be regenerated at measurement time)
    "pre_period_trends",       # Parallel trends validation
    "model_input_timeseries",  # Full time series fed to model
    "model_fit_stats",         # Model diagnostic statistics
    "daily_effects",           # CRITICAL: Day-by-day effects for charting
    "impact_summary",          # Headline results
    "model_residuals",         # Residuals across periods
    "robustness_checks"        # Sensitivity analysis
]


# =============================================================================
# VALIDATION FILE CREATION FUNCTIONS
# =============================================================================

def create_validation_file(test_id: str, file_type: str, data: Optional[pd.DataFrame] = None) -> str:
    """
    Create a validation CSV file for a test.
    
    Args:
        test_id: The test identifier
        file_type: One of the validation file types (e.g., 'pre_period_summary')
        data: Optional DataFrame with data to populate the file
        
    Returns:
        Path to the created file
    """
    if file_type not in VALIDATION_FILE_SCHEMAS:
        raise ValueError(f"Unknown file type: {file_type}. Valid types: {list(VALIDATION_FILE_SCHEMAS.keys())}")
    
    validation_folder = os.path.join(TEST_VALIDATION_DIR, f"{test_id}_test_validation_files")
    os.makedirs(validation_folder, exist_ok=True)
    
    file_path = os.path.join(validation_folder, f"{test_id}_{file_type}.csv")
    columns = VALIDATION_FILE_SCHEMAS[file_type]
    
    if data is not None:
        # Use provided data, ensuring columns match
        df = data[columns] if all(c in data.columns for c in columns) else data
    else:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=columns)
    
    df.to_csv(file_path, index=False)
    logger.info(f"Created validation file: {file_path}")
    
    return file_path


def create_design_phase_files(test_id: str, pre_period_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
    """
    Create all validation files for the test design finalization phase.
    
    Args:
        test_id: The test identifier
        pre_period_data: Optional dict with DataFrames for each file type
        
    Returns:
        List of created file paths
    """
    created_files = []
    
    for file_type in DESIGN_PHASE_FILES:
        data = pre_period_data.get(file_type) if pre_period_data else None
        file_path = create_validation_file(test_id, file_type, data)
        created_files.append(file_path)
    
    return created_files


def create_measurement_phase_files(test_id: str, measurement_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
    """
    Create all validation files for the measurement completion phase.
    
    Args:
        test_id: The test identifier
        measurement_data: Optional dict with DataFrames for each file type
        
    Returns:
        List of created file paths
    """
    created_files = []
    
    for file_type in MEASUREMENT_PHASE_FILES:
        data = measurement_data.get(file_type) if measurement_data else None
        file_path = create_validation_file(test_id, file_type, data)
        created_files.append(file_path)
    
    return created_files


# =============================================================================
# VERSION HISTORY FUNCTIONS
# =============================================================================

def create_version_history_entry(test_id: str, action: str, details: Dict[str, Any]) -> str:
    """
    Create or append to a version history file for a test.
    
    Args:
        test_id: The test identifier
        action: Action performed (e.g., 'created', 'updated', 'measured')
        details: Dict with details about the action
        
    Returns:
        Path to the version history file
    """
    version_folder = os.path.join(VERSION_HISTORY_DIR, f"{test_id}_version_history")
    os.makedirs(version_folder, exist_ok=True)
    
    version_num = get_next_version_number(test_id)
    file_path = os.path.join(version_folder, f"{test_id}_version_history_{version_num}.csv")
    
    entry = {
        "version": version_num,
        "timestamp": datetime.now().isoformat(),
        "action": action,
        **details
    }
    
    df = pd.DataFrame([entry])
    df.to_csv(file_path, index=False)
    
    logger.info(f"Created version history entry: {file_path}")
    
    return file_path


# =============================================================================
# MASTER TESTING DOC FUNCTIONS
# =============================================================================

def generate_test_id() -> str:
    """
    Generate a unique test ID based on timestamp.
    
    Returns:
        Test ID like 'TEST-MMDDHHSS'
    """
    now = datetime.now()
    return f"TEST-{now.strftime('%m%d%H%M')}"


def add_test_to_master(test_config: Dict[str, Any]) -> str:
    """
    Add a new test to the master testing document.
    
    This is called when a user confirms they are aligned with the test design.
    
    Args:
        test_config: Dict with test configuration including:
            - test_name, test_description, start_date, end_date
            - split_method, test_dmas (or test_cust_count, test_audience_list)
            - pre_period_lookback_start, pre_period_lookback_end
            
    Returns:
        The generated test_id
    """
    # Generate test ID
    test_id = generate_test_id()
    
    # Create folders
    folders = create_test_folders(test_id)
    
    # Prepare row data
    now = datetime.now()
    row_data = {
        "test_id": test_id,
        "test_name": test_config.get("test_name", ""),
        "status": "Scheduled",
        "test_description": test_config.get("test_description", ""),
        "start_date": test_config.get("start_date", ""),
        "end_date": test_config.get("end_date", ""),
        "split_method": test_config.get("split_method", ""),
        "test_dmas": test_config.get("test_dmas", "not applicable"),
        "test_cust_count": test_config.get("test_cust_count", "not applicable"),
        "test_audience_list": f"{test_id}_test_user_list" if test_config.get("split_method") == "customer" else "not applicable",
        "measurement_method": "causal impact",
        "pre_period_lookback_start": test_config.get("pre_period_lookback_start", ""),
        "pre_period_lookback_end": test_config.get("pre_period_lookback_end", ""),
        "{TEST_ID KEY}_test_validation_files": f"{test_id}_test_validation_files",
        "avg_lift": "",
        "cum_lift": "",
        "cred_int_lower": "",
        "cred_int_upper": "",
        "prob_lift_gt_0": "",
        "p_value_equivalent": "",
        "relative_lift_pct": "",
        "measurement_run_date": "",
        "created_at": now.strftime("%Y-%m-%d %H:%M"),
        "last_modified_at": now.strftime("%Y-%m-%d %H:%M")
    }
    
    # Load existing master doc or create new
    if os.path.exists(MASTER_TESTING_DOC):
        df = pd.read_csv(MASTER_TESTING_DOC)
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])
    
    df.to_csv(MASTER_TESTING_DOC, index=False)
    
    # Create version history entry
    create_version_history_entry(test_id, "created", test_config)
    
    logger.info(f"Added test {test_id} to master testing document")
    
    return test_id


def update_test_status(test_id: str, new_status: str) -> bool:
    """
    Update the status of a test in the master document.
    
    Args:
        test_id: The test identifier
        new_status: New status (Scheduled/Running/Complete)
        
    Returns:
        True if successful
    """
    if not os.path.exists(MASTER_TESTING_DOC):
        return False
    
    df = pd.read_csv(MASTER_TESTING_DOC)
    mask = df['test_id'] == test_id
    
    if not mask.any():
        return False
    
    df.loc[mask, 'status'] = new_status
    df.loc[mask, 'last_modified_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(MASTER_TESTING_DOC, index=False)
    
    # Create version history entry
    create_version_history_entry(test_id, "status_updated", {"new_status": new_status})
    
    return True


def update_test_with_measurement_results(test_id: str, results: Dict[str, Any]) -> bool:
    """
    Update a test with causal impact measurement results.
    
    This is called when measurement is complete and signed off by user.
    
    Args:
        test_id: The test identifier
        results: Dict with causal impact results:
            - avg_lift, cum_lift, cred_int_lower, cred_int_upper
            - prob_lift_gt_0, p_value_equivalent, relative_lift_pct
            
    Returns:
        True if successful
    """
    if not os.path.exists(MASTER_TESTING_DOC):
        return False
    
    df = pd.read_csv(MASTER_TESTING_DOC)
    mask = df['test_id'] == test_id
    
    if not mask.any():
        return False
    
    now = datetime.now()
    
    # Update measurement results
    df.loc[mask, 'avg_lift'] = results.get('avg_lift', '')
    df.loc[mask, 'cum_lift'] = results.get('cum_lift', '')
    df.loc[mask, 'cred_int_lower'] = results.get('cred_int_lower', '')
    df.loc[mask, 'cred_int_upper'] = results.get('cred_int_upper', '')
    df.loc[mask, 'prob_lift_gt_0'] = results.get('prob_lift_gt_0', '')
    df.loc[mask, 'p_value_equivalent'] = results.get('p_value_equivalent', '')
    df.loc[mask, 'relative_lift_pct'] = results.get('relative_lift_pct', '')
    df.loc[mask, 'measurement_run_date'] = now.strftime("%Y-%m-%d")
    df.loc[mask, 'status'] = 'Complete'
    df.loc[mask, 'last_modified_at'] = now.strftime("%Y-%m-%d %H:%M")
    
    df.to_csv(MASTER_TESTING_DOC, index=False)
    
    # Create version history entry
    create_version_history_entry(test_id, "measurement_complete", results)
    
    logger.info(f"Updated test {test_id} with measurement results")
    
    return True


def get_test_by_id(test_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a test's details from the master document.
    
    Args:
        test_id: The test identifier
        
    Returns:
        Dict with test details or None if not found
    """
    if not os.path.exists(MASTER_TESTING_DOC):
        return None
    
    df = pd.read_csv(MASTER_TESTING_DOC)
    mask = df['test_id'] == test_id
    
    if not mask.any():
        return None
    
    return df[mask].iloc[0].to_dict()


def get_all_tests() -> pd.DataFrame:
    """
    Get all tests from the master document.
    
    Returns:
        DataFrame with all tests
    """
    if not os.path.exists(MASTER_TESTING_DOC):
        return pd.DataFrame()
    
    return pd.read_csv(MASTER_TESTING_DOC)
