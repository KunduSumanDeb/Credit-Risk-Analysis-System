# Credit Risk Analysis System
## MEMORY.md

---

# Project Overview

Project Name:
Credit Risk Analysis System

Project Type:
Data Science Capstone Project

Goal:
Develop an end-to-end machine learning system capable of predicting loan default risk while providing explainable predictions and actionable recommendations through an interactive dashboard.

---

# Project Architecture

The project follows a modular architecture.

```
Raw Data
      │
      ▼
Preprocessing
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
```

Each module is implemented independently.

Business logic remains inside the **src** directory.

Jupyter notebooks are only responsible for executing functions and documenting the workflow.

---

# Folder Structure

```
dashboard/
data/
deployment/
notebooks/
outputs/
reports/
src/
tests/
```

---

# Development Principles

- Keep every module independent.
- Avoid duplicate code.
- Follow reusable function design.
- Keep notebooks clean.
- Store reusable logic inside src.
- Every function should have a complete NumPy-style docstring.
- Follow PEP-8 formatting.
- Use descriptive variable names.
- Avoid hardcoded values.
- Configuration values belong in config.py or constants.py.

---

# Module Responsibilities

## preprocessing.py

Responsible for:

- Loading datasets
- Validation
- Duplicate handling
- Missing value handling
- Data cleaning
- Export cleaned dataset

---

## visualization.py

Responsible for:

- Dataset overview
- Target analysis
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Outlier analysis
- Figure generation

---

## feature_engineering.py

Responsible for:

- Feature selection
- Encoding categorical variables
- Scaling numerical variables
- Feature transformation
- Dataset splitting
- Export processed datasets

---

## model_training.py

Responsible for:

- Training machine learning models
- Hyperparameter tuning
- Cross-validation
- Model persistence

---

## evaluation.py

Responsible for:

- Classification metrics
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Performance comparison

---

## explainability.py

Responsible for:

- Feature importance
- SHAP analysis
- Global explanations
- Local explanations

---

## recommendation.py

Responsible for:

- Risk recommendations
- Customer suggestions
- Loan decision assistance

---

## prediction.py

Responsible for:

- Load trained model
- Accept new customer data
- Return prediction
- Return probability
- Return explanation

---

## utils.py

Contains reusable helper functions shared across modules.

---

# Coding Standards

Always

- Write modular functions.
- Return objects instead of printing whenever possible.
- Keep plotting and computation separate.
- Validate function inputs.
- Raise meaningful exceptions.
- Use type hints.
- Keep comments concise.

Avoid

- Global variables
- Magic numbers
- Repeated logic
- Long notebook cells
- Mixing preprocessing with visualization

---

# Current Progress

## Module 1
Status: Completed

Completed:

- Dataset loading
- Validation
- Duplicate removal
- Missing value treatment
- Clean dataset export

---

## Module 2
Status: Completed

Completed:

- Dataset overview
- Target analysis
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Outlier analysis

---

## Module 3
Status: Ready to Begin

Next Tasks

- Feature selection
- Feature encoding
- Feature scaling
- Train/Validation/Test split
- Export processed datasets

---

# Machine Learning Pipeline

```
Raw Data
    ↓
Cleaning
    ↓
EDA
    ↓
Feature Engineering
    ↓
Training
    ↓
Evaluation
    ↓
Explainability
    ↓
Recommendation
    ↓
Prediction
```

---

# Important Rules

- Never edit raw datasets.
- Always save intermediate datasets.
- Save trained models inside data/models.
- Save processed datasets inside data/processed.
- Save figures inside outputs/figures.
- Save reports inside outputs/reports.
- Keep notebooks reproducible from the first cell.
- Restart kernel and run all before committing changes.

---

# Git Workflow

Before every commit:

- Run all notebook cells.
- Verify outputs.
- Remove temporary debugging code.
- Update MEMORY.md if architecture changes.
- Push only clean and reproducible code.

---

Last Updated

Module 2 (Exploratory Data Analysis) Completed.

Next Module:
Feature Engineering