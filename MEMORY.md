# Credit Risk Analysis System

## MEMORY.md

---

# Project Overview

**Project Name:**
Credit Risk Analysis System

**Project Type:**
Data Science Capstone Project

**Goal:**
Develop an end-to-end machine learning system capable of predicting loan default risk while providing explainable predictions and actionable recommendations through an interactive Streamlit dashboard.

---

# Project Architecture

The project follows a modular, production-oriented architecture.

```
Raw Data
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Explainability
      │
      ▼
Recommendation Engine
      │
      ▼
Prediction Module
      │
      ▼
Dashboard
      │
      ▼
Deployment
```

Each module is implemented independently.

Business logic resides inside **src/**.

Jupyter notebooks only execute functions and document the workflow.

---

# Folder Structure

```
Credit-Risk-Analysis-System/
│
├── dashboard/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── models/
│
├── deployment/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── reports/
│
├── reports/
│
├── src/
│
└── tests/
```

---

# Development Principles

- Modular software architecture
- Low coupling
- High cohesion
- Single Responsibility Principle
- One responsibility per function
- Reusable business logic
- Notebook contains no reusable business logic
- NumPy-style docstrings
- Type hints
- PEP-8 formatting
- Descriptive variable names
- Configuration through `config.py`
- Constants through `constants.py`
- Return values instead of printing from reusable functions
- Separate computation from visualization
- Validate inputs before processing
- Raise meaningful exceptions
- Avoid hardcoded paths
- Avoid magic numbers
- Avoid duplicated business logic
- Preserve compatibility between modules
- Prefer generic reusable functions where appropriate

---

# Notebook Philosophy

Jupyter notebooks are used for:

- Demonstrating module usage
- Executing functions
- Verification
- Documentation
- Visualization
- Recording expected outputs

Reusable business logic must **not** be implemented directly inside notebooks.

The notebook should read as a reproducible workflow rather than as a collection of unrelated cells.

**Typical structure:**

```
Load
  ↓
Validate
  ↓
Process
  ↓
Transform
  ↓
Execute
  ↓
Verify
  ↓
Export
```

Every notebook should be executable from a fresh kernel.

**Before committing:**

```
Restart Kernel
      ↓
Run All Cells
      ↓
Verify Output
```

---

# Function Development Workflow

Every new function must follow this sequence.

### Step 1 — Design Discussion

Before implementation, discuss:

- Purpose
- Responsibility
- Inputs
- Outputs
- Return type
- Validation
- Exceptions
- Edge cases
- Integration with existing modules
- Possible architectural alternatives
- Trade-offs

Do not immediately write code when the architecture is ambiguous.

### Step 2 — Complete Implementation

Every function must contain:

- Complete implementation
- Type hints
- NumPy-style docstring
- Parameters section
- Returns section
- Raises section
- Input validation
- Meaningful exceptions
- Appropriate comments
- Explicit return statement

Never provide partial function snippets when modifying an existing function.

### Step 3 — Notebook Integration

Every function added to a module should have corresponding notebook usage, containing:

- Markdown cell
- Import cell
- Execution cell
- Verification cell
- Expected output

### Step 4 — Testing

Each function should be tested with:

- Normal input
- Invalid input
- Empty input where applicable
- Edge cases
- Expected output

### Step 5 — Final Verification

After implementation, verify:

- Function output
- Validation behavior
- Architecture
- Integration
- Coding style
- Fresh-kernel execution
- Compatibility with previous modules

---

# Configuration Philosophy

Configuration values must be centralized inside:

`src/config.py`

Avoid hardcoding:

- Paths
- Random states
- Dataset names
- Target column
- Train/validation/test split sizes
- Model artifact paths
- Other project-level configuration values

**Examples of configuration values include:**

- `PROJECT_ROOT`
- `DATA_DIR`
- `RAW_DATA_DIR`
- `INTERIM_DATA_DIR`
- `PROCESSED_DATA_DIR`
- `MODEL_DIR`
- `TARGET_COLUMN`
- `RANDOM_STATE`
- `TEST_SIZE`
- `VALIDATION_SIZE`
- `TRAINING_ARTIFACTS_PATH`

---

# Persistence Architecture

The project uses pickle persistence for reusable Python objects where appropriate.

Reusable persistence functions are implemented in:

`src/feature_engineering.py`

**Functions:**

- `save_pickle()`
- `load_pickle()`

Module 3 persists the processed training artifacts so that Module 4 does not depend on variables remaining in the Module 3 notebook kernel.

**Configured artifact path:**

`data/processed/training_artifacts.pkl`

**Configuration constant:**

`TRAINING_ARTIFACTS_PATH`

**The training artifact contains:**

- `X_train`
- `X_validation`
- `X_test`
- `y_train`
- `y_validation`
- `y_test`

This provides a clean handoff between Feature Engineering and Model Training.

---

# Module Responsibilities

## preprocessing.py

Responsible for:

- Dataset loading
- Dataset validation
- Duplicate detection
- Duplicate removal
- Missing value analysis
- Missing value treatment
- Dataset cleaning
- Export cleaned dataset

---

## visualization.py

Responsible for:

- Dataset overview
- Target distribution
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Outlier analysis
- Figure generation

Visualization logic remains separate from computational/business logic.

---

## feature_engineering.py

**Validation**
- `_validate_dataframe()`
- `_validate_columns_exist()`

**Feature Discovery**
- `get_numerical_columns()`
- `get_categorical_columns()`

**Dataset Preparation**
- `split_features_target()`
- `split_dataset()`

**Preprocessing Pipeline**
- `build_preprocessor()`
- `fit_preprocessor()`
- `transform_dataset()`

**Persistence**
- `save_pickle()`
- `load_pickle()`

Responsible for:

- Feature identification
- Feature selection
- Train / Validation / Test split
- Stratified sampling
- Numerical scaling
- Categorical encoding
- Building preprocessing pipeline
- Fitting preprocessing pipeline
- Transforming datasets into ML-ready format
- Preserving feature names
- Preserving DataFrame structure
- Persisting processed artifacts

---

## model_training.py

Responsible for:

- Model creation
- Model training
- Cross validation
- Hyperparameter tuning
- Model persistence

---

## evaluation.py

Responsible for:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision Recall Curve
- Model comparison

---

## explainability.py

Responsible for:

- Feature importance
- Permutation importance
- SHAP
- Global explanation
- Local explanation

---

## recommendation.py

Responsible for:

- Risk recommendations
- Customer suggestions
- Loan approval assistance

---

## prediction.py

Responsible for:

- Load trained model
- Load preprocessing pipeline
- Accept customer input
- Predict default probability
- Return recommendation
- Return explanation

---

## utils.py

Reusable helper functions shared across modules.

---

# Coding Standards

**Always**

- Small reusable functions
- Single Responsibility Principle
- Return objects instead of printing
- Validate function inputs
- Raise meaningful exceptions
- Use type hints
- NumPy docstrings
- PEP-8 formatting
- Keep plotting separate from computation

**Avoid**

- Hardcoded values
- Hardcoded paths
- Duplicate code
- Business logic inside notebooks
- Large notebook cells
- Global variables
- Magic numbers

---

# Module Progress

## Module 1 — Data Preprocessing

**Status:** ✅ COMPLETED

Implemented:

- Dataset loading
- Dataset validation
- Duplicate detection
- Duplicate removal
- Missing value analysis
- Missing value treatment
- Dataset cleaning
- Clean dataset export

---

## Module 2 — Exploratory Data Analysis

**Status:** ✅ COMPLETED

Implemented:

- Dataset overview
- Target variable analysis
- Numerical feature analysis
- Categorical feature analysis
- Correlation analysis
- Outlier analysis
- Feature vs Target analysis
- Visualization functions
- Outlier detection

---

## Module 3 — Feature Engineering

**Status:** ✅ COMPLETED

**Validation**
- `_validate_dataframe()`
- `_validate_columns_exist()`

**Feature Discovery**
- `get_numerical_columns()`
- `get_categorical_columns()`

**Dataset Preparation**
- `split_features_target()`
- `split_dataset()`

The dataset is divided into Train / Validation / Test using stratified sampling.

**Preprocessing Pipeline**
- `build_preprocessor()` — uses `StandardScaler()`, `OneHotEncoder()`, `ColumnTransformer()`, `Pipeline()`

**Preprocessor Fitting**
- `fit_preprocessor()` — fitted only on `X_train` to avoid data leakage

**Dataset Transformation**
- `transform_dataset()` — returns pandas DataFrames instead of raw NumPy arrays

Benefits:
- Preserves feature names
- Preserves indices
- Easier debugging
- Better explainability
- Dashboard-friendly output

**Verified:**

- Dataset shapes
- Missing values
- Feature names
- Target distribution
- StandardScaler
- OneHotEncoder
- Processed DataFrame output
- Original index preservation
- Train/validation/test consistency

**Artifact Export**

The processed datasets and target splits are persisted into:

`data/processed/training_artifacts.pkl`

```python
{
    "X_train": X_train_processed,
    "X_validation": X_validation_processed,
    "X_test": X_test_processed,
    "y_train": y_train,
    "y_validation": y_validation,
    "y_test": y_test,
}
```

This artifact is loaded independently by Module 4.

---

## Module 4 — Model Training

**Status:** ✅ COMPLETED

Module 4 is fully implemented and verified.

### Architecture

```
Validation
      ↓
Model Builders
      ↓
Generic Training
      ↓
Class Prediction
      ↓
Probability Prediction
      ↓
Integration Verification
```

The project intentionally avoids model-specific training functions such as `train_logistic_regression()`, `train_decision_tree()`, `train_random_forest()`.

Instead, all models use the generic `train_model()` interface.

### Validation

Implemented: `_validate_model()`

Responsible for validating that the supplied object is a valid scikit-learn estimator.

### Model Builders

Implemented:

- `build_logistic_regression()`
- `build_decision_tree()`
- `build_random_forest()`

The builders:
- Create untrained estimators
- Use the configured random state
- Support configurable model parameters
- Avoid model-specific training functions
- Return model objects

### Generic Training

Implemented: `train_model()`

Responsibilities:
- Validate model
- Validate `X_train`
- Validate `y_train`
- Reject empty training datasets
- Validate matching feature/target lengths
- Fit the supplied model
- Return the fitted model

The function is model-agnostic.

### Prediction

Implemented: `predict_model()`

Responsibilities:
- Validate model
- Validate feature dataset
- Reject empty prediction datasets
- Verify that the model is fitted
- Generate class predictions

Returns: `numpy.ndarray`

### Probability Prediction

Implemented: `predict_probability()`

Responsibilities:
- Validate model
- Validate feature dataset
- Reject empty prediction datasets
- Verify fitted model
- Verify probability prediction support
- Generate class probabilities

Returns a two-dimensional NumPy array:
- Column 0 → Probability of class 0
- Column 1 → Probability of class 1

For the credit-risk problem: Class 0 → No Default, Class 1 → Default.

Therefore `probabilities[:, 1]` represents the predicted default probability.

### Validation Tests

**Generic Training**
- Valid model training
- Invalid model
- Invalid X_train
- Invalid y_train
- Empty X_train
- Empty y_train
- Mismatched X_train / y_train lengths

**Prediction**
- Valid prediction
- Invalid prediction data
- Empty prediction data
- Unfitted model

**Probability Prediction**
- Valid probability prediction
- Correct probability shape
- Probability range validation
- Probabilities summing to 1
- Invalid prediction data
- Empty prediction data
- Unfitted model

### Three Model Integration

All three models successfully passed the generic pipeline:

```
Build Model
     ↓
train_model()
     ↓
predict_model()
     ↓
predict_probability()
```

Verified:

- Logistic Regression ✅
- Decision Tree ✅
- Random Forest ✅

**✅ ALL THREE MODELS PASSED INTEGRATION VERIFICATION**

### Fresh Kernel Verification

The complete Module 4 notebook was tested using Restart Kernel → Run All Cells, executing successfully from the beginning through final integration verification.

This confirms:

- Module 4 does not depend on the Module 3 notebook kernel
- Training artifacts are correctly persisted
- Training artifacts are correctly loaded
- Required imports are available
- Models can be built independently
- Models can be trained independently
- Predictions can be generated independently
- Probability predictions can be generated independently
- All three models pass integration verification

### Issues Resolved

**Issue 1 — Missing Training Variables**

Problem: `X_train` / `y_train` were dependent on the previous notebook's kernel state.

Solution: `training_artifacts.pkl` was introduced as a persistent handoff between Module 3 and Module 4.

**Issue 2 — Missing pandas Import**

Problem: `model_training.py` used `pd.DataFrame` / `pd.Series` during validation but did not initially import pandas.

Solution: `import pandas as pd` was added to the source module.

**Issue 3 — Missing NumPy Import in Notebook**

Problem: A verification cell used `np.ndarray` without explicitly importing NumPy in the notebook section.

Solution: `import numpy as np` was added to the notebook's prediction section.

**Development Lesson:** A notebook must not rely on hidden state from previously executed cells. Source modules must explicitly declare their dependencies.

---

## Module 5 — Model Evaluation

**Status:** ➡ NEXT (Current)

**Primary source module:** `src/evaluation.py`

**Notebook:** `notebooks/05_model_evaluation.ipynb`

The evaluation module must remain separate from `model_training.py`.

### Responsibilities

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve
- Model Comparison

Evaluation should consume outputs from Module 4 rather than retraining models.

### Planned Architecture

```
Trained Models
      │
      ├───────────────┐
      │               │
      ▼               ▼
Predicted Classes   Probabilities
      │               │
      └───────┬───────┘
              ▼
        Evaluation
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
   Metrics  Curves  Reports
              │
              ▼
       Model Comparison
```

### Design Principles

Evaluation functions should:

- Accept model outputs rather than retraining models
- Validate input types
- Validate matching lengths
- Reject empty inputs
- Avoid hardcoded target values where configuration is appropriate
- Return structured results
- Keep visualization separate from metric computation
- Remain model-agnostic
- Avoid duplicated evaluation logic

---

# Future Modules

### Module 6 — Explainability

Responsible for:

- Feature importance
- Permutation importance
- SHAP
- Global explanation
- Local explanation

### Module 7 — Recommendation Engine

Responsible for:

- Risk recommendations
- Customer suggestions
- Loan approval assistance

### Module 8 — Prediction Module

Responsible for:

- Loading trained model
- Loading preprocessing pipeline
- Accepting customer input
- Predicting default probability
- Returning recommendation
- Returning explanation

### Dashboard

Responsible for the interactive Streamlit interface.

---

# Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Explainability
      │
      ▼
Recommendation Engine
      │
      ▼
Prediction
      │
      ▼
Dashboard
      │
      ▼
Deployment
```

---

# Data Storage Rules

- Never modify raw datasets.
- Use `data/raw/` for original datasets.
- Use `data/interim/` for cleaned/intermediate datasets.
- Use `data/processed/` for ML-ready datasets and preprocessing/training artifacts.
- Use `data/models/` for trained model artifacts.
- Use `outputs/figures/` for generated figures.
- Use `outputs/metrics/` for evaluation metrics.
- Use `outputs/predictions/` for prediction outputs.
- Use `outputs/reports/` for generated reports.

---

# Git Workflow

Before every commit:

1. Restart notebook kernel.
2. Run all notebook cells.
3. Verify outputs.
4. Remove debugging code.
5. Verify folder structure.
6. Check source files.
7. Update MEMORY.md if architecture changes.
8. Commit clean reproducible code.

---

# Important Architectural Rules

- Do not rewrite completed modules unless a genuine issue is discovered.
- Do not introduce architecture changes halfway through a module.
- Discuss architectural changes before implementing them.
- Do not duplicate existing helper functions.
- Do not place reusable business logic inside notebooks.
- Do not use notebook variables as hidden dependencies.
- Persist artifacts required by downstream modules.
- Keep imports explicit.
- Keep modules independently understandable.
- Preserve compatibility with completed modules.
- Prefer generic reusable functions.
- Keep evaluation separate from model training.
- Keep visualization separate from computation.
- Follow the design → implementation → testing → verification workflow.

---

# Current Project Status

**Completed:**

- ✅ Module 1 — Data Preprocessing
- ✅ Module 2 — Exploratory Data Analysis
- ✅ Module 3 — Feature Engineering
- ✅ Module 4 — Model Training

**Current:**

- ➡ Module 5 — Model Evaluation

**Next primary file:** `src/evaluation.py`

**Next notebook:** `notebooks/05_model_evaluation.ipynb`

---

# Last Updated

**Completed Modules:**

- ✅ Module 1 — Data Preprocessing
- ✅ Module 2 — Exploratory Data Analysis
- ✅ Module 3 — Feature Engineering
- ✅ Module 4 — Model Training

**Current Module:**

- ➡ Module 5 — Model Evaluation