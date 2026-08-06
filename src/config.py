# Configuration utilities
"""
Project Configuration
=====================

This module contains all configurable paths and project settings
used throughout the Credit Risk Analysis System.
"""

from pathlib import Path

# ============================================================================
# Project Root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# Data Directories
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# ============================================================================
# Output Directories
# ============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"

# ============================================================================
# Dashboard
# ============================================================================

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# ============================================================================
# Dataset Files
# ============================================================================

CREDIT_RISK_DATASET = RAW_DATA_DIR / "credit_risk_dataset.csv"
BANK_DATASET = RAW_DATA_DIR / "bank.csv"
LOAN_DATASET = RAW_DATA_DIR / "loan_data.csv"

# ============================================================================
# Model Files
# ============================================================================

BEST_MODEL = MODEL_DIR / "best_model.pkl"
SCALER = MODEL_DIR / "scaler.pkl"
ENCODER = MODEL_DIR / "encoder.pkl"
LABEL_ENCODER = MODEL_DIR / "label_encoder.pkl"