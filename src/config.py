"""
Configuration utilities.

This module contains all configurable paths and project settings
used throughout the Credit Risk Analysis System.
"""

from pathlib import Path

# =============================================================================
# Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# Data Directories
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# =============================================================================
# Output Directories
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"

# =============================================================================
# Dashboard Directory
# =============================================================================

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# =============================================================================
# Raw Dataset Files
# =============================================================================

CREDIT_RISK_DATASET_PATH = RAW_DATA_DIR / "credit_risk_dataset.csv"
BANK_DATASET_PATH = RAW_DATA_DIR / "bank.csv"
LOAN_DATASET_PATH = RAW_DATA_DIR / "loan_data.csv"

# =============================================================================
# Processed Dataset Files
# =============================================================================

CLEANED_DATA_PATH = INTERIM_DATA_DIR / "cleaned_data.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# =============================================================================
# Model Artifact Paths
# =============================================================================

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# =============================================================================
# Feature Engineering Configuration
# =============================================================================

TARGET_COLUMN = "loan_status"

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20

# =============================================================================
# Export Configuration
# =============================================================================

EXPORT_INDEX = False

# =============================================================================
# Model Training Configuration
# =============================================================================

CROSS_VALIDATION_FOLDS = 5

N_JOBS = -1

VERBOSE = 1