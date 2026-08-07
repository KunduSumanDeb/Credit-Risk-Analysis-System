# Credit Risk Analysis System

## MEMORY.md

---

# Project Overview

Project Name:
Credit Risk Analysis System

Project Type:
Data Science Capstone Project

Goal:
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
dashboard/
data/
│
├── raw/
├── interim/
├── processed/
└── models/
│
deployment/
notebooks/
outputs/
│
├── figures/
├── metrics/
├── predictions/
└── reports/
│
reports/
src/
tests/
```

---

# Development Principles

- Modular software architecture
- Low coupling, high cohesion
- One responsibility per function
- Reusable business logic
- Notebook contains no reusable code
- NumPy style docstrings
- Type hints
- PEP-8 formatting
- Descriptive variable names
- Configuration through config.py
- Constants through constants.py
- Return values instead of printing
- Separate computation from visualization
- Validate inputs before processing
- Raise meaningful exceptions

---

# Module Responsibilities

## preprocessing.py

Responsible for

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

Responsible for

- Dataset overview
- Target distribution
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Outlier analysis
- Figure generation

---

## feature_engineering.py

Responsible for

### Validation

- _validate_dataframe()
- _validate_columns_exist()

### Feature Discovery

- get_numerical_columns()
- get_categorical_columns()

### Dataset Preparation

- split_features_target()
- split_dataset()

### Preprocessing Pipeline

- build_preprocessor()
- fit_preprocessor()
- transform_dataset()

Responsible for

- Feature identification
- Feature selection
- Train / Validation / Test split
- Numerical scaling
- Categorical encoding
- Building preprocessing pipeline
- Fitting preprocessing pipeline
- Transforming datasets into ML-ready format

---

## model_training.py

Responsible for

- Model creation
- Model training
- Cross validation
- Hyperparameter tuning
- Model persistence

---

## evaluation.py

Responsible for

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

Responsible for

- Feature importance
- Permutation importance
- SHAP
- Global explanation
- Local explanation

---

## recommendation.py

Responsible for

- Risk recommendations
- Customer suggestions
- Loan approval assistance

---

## prediction.py

Responsible for

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

Always

- Small reusable functions
- Single Responsibility Principle
- Return objects instead of printing
- Validate function inputs
- Raise meaningful exceptions
- Use type hints
- NumPy docstrings
- PEP-8 formatting
- Keep plotting separate from computation

Avoid

- Hardcoded values
- Hardcoded paths
- Duplicate code
- Business logic inside notebooks
- Large notebook cells
- Global variables
- Magic numbers

---

# Current Progress

## Module 1

Status:
Completed

Completed

- Dataset loading
- Dataset validation
- Duplicate removal
- Missing value treatment
- Dataset cleaning
- Export cleaned dataset

---

## Module 2

Status:
Completed

Completed

- Dataset overview
- Target distribution
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Outlier analysis
- Feature vs Target analysis
- Outlier detection

---

## Module 3

Status:
Completed

Completed

### Validation

- _validate_dataframe()
- _validate_columns_exist()

### Feature Discovery

- Numerical column identification
- Categorical column identification

### Dataset Preparation

- Feature / Target separation
- Stratified Train / Validation / Test split

### Feature Engineering

- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline construction
- Pipeline fitting
- Dataset transformation

### Verification

- Processed dataset validation
- Feature name preservation
- DataFrame output
- Original index preservation
- Target distribution verification

---

## Module 4

Status:
Ready to Begin

Planned Tasks

- Build Logistic Regression
- Build Decision Tree
- Build Random Forest
- Generic model training function
- Prediction function
- Probability prediction
- Cross validation
- Hyperparameter tuning
- Best model selection

---

# Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Cleaning
      │
      ▼
EDA
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
```

---

# Important Rules

- Never modify raw datasets.
- Store cleaned datasets inside data/interim.
- Store processed datasets inside data/processed.
- Store trained models inside data/models.
- Store figures inside outputs/figures.
- Store reports inside outputs/reports.
- Keep notebooks reproducible.
- Restart kernel and Run All before committing.
- Maintain modular architecture.
- Every notebook should only demonstrate usage.

---

# Git Workflow

Before every commit

- Run all notebook cells
- Verify outputs
- Remove debugging code
- Verify folder structure
- Update MEMORY.md if architecture changes
- Commit clean reproducible code

---

# Last Updated

Modules Completed

- ✅ Module 1 — Data Preprocessing
- ✅ Module 2 — Exploratory Data Analysis
- ✅ Module 3 — Feature Engineering

Next Module

➡ Module 4 — Model Training