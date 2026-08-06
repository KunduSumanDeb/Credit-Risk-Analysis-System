# Data preprocessing functions
"""
Data Preprocessing Module
=========================

This module contains functions for loading, inspecting,
validating, and preprocessing the credit risk dataset.
"""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    ValueError
        If the dataset is empty.

    Exception
        For any unexpected loading error.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    try:
        dataframe = pd.read_csv(file_path)

        if dataframe.empty:
            raise ValueError(
                "The dataset is empty."
            )

        return dataframe

    except Exception as error:
        raise Exception(
            f"Error loading dataset: {error}"
        ) from error

def get_dataset_shape(dataframe: pd.DataFrame) -> tuple[int, int]:
    """
    Get the dimensions of the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    tuple[int, int]
        Number of rows and columns.
    """

    return dataframe.shape

def display_sample_records(
    dataframe: pd.DataFrame,
    n: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the first and last n records.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    n : int, default=5
        Number of rows to display.

    Returns
    -------
    tuple
        Head and tail of the dataset.
    """

    return dataframe.head(n), dataframe.tail(n)

def get_column_information(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate column-wise information.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Summary of dataset columns.
    """

    information = pd.DataFrame({
        "Column": dataframe.columns,
        "Data Type": dataframe.dtypes.values,
        "Non-Null Count": dataframe.count().values,
        "Missing Values": dataframe.isnull().sum().values,
        "Unique Values": dataframe.nunique().values
    })

    return information

# ============================================================================
# Missing Value Analysis
# ============================================================================

def check_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Missing value count and percentage for columns
        containing missing values only.
    """

    missing_summary = pd.DataFrame({
        "Missing Values": dataframe.isnull().sum(),
        "Missing Percentage": (
            dataframe.isnull().mean() * 100
        ).round(2)
    })

    missing_summary = missing_summary[
        missing_summary["Missing Values"] > 0
    ]

    return missing_summary.sort_values(
        by="Missing Values",
        ascending=False
    )

# ============================================================================
# Duplicate Analysis
# ============================================================================

def check_duplicate_records(dataframe: pd.DataFrame) -> int:
    """
    Count duplicate rows in the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    int
        Number of duplicate records.
    """

    return dataframe.duplicated().sum()

# ============================================================================
# Memory Analysis
# ============================================================================

def get_memory_usage(dataframe: pd.DataFrame) -> float:
    """
    Calculate memory usage of the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    float
        Memory usage in megabytes.
    """

    memory = (
        dataframe.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    return round(memory, 2)

# ============================================================================
# Dataset Summary
# ============================================================================

def generate_dataset_summary(
    dataframe: pd.DataFrame
) -> dict:
    """
    Generate an overall dataset summary.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Dataset summary.
    """

    rows, columns = get_dataset_shape(dataframe)

    summary = {
        "Rows": int(rows),
        "Columns": int(columns),
        "Numerical Columns": int(
            dataframe.select_dtypes(include="number").shape[1]
        ),
        "Categorical Columns": int(
            dataframe.select_dtypes(exclude="number").shape[1]
        ),
        "Duplicate Rows": int(
            check_duplicate_records(dataframe)
        ),
        "Memory Usage (MB)": float(
            get_memory_usage(dataframe)
        )
    }

    return summary

# ============================================================================
# Numerical Summary
# ============================================================================

def get_numerical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics of numerical columns.
    """

    return dataframe.describe().T

# ============================================================================
# Categorical Summary
# ============================================================================

def get_categorical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for categorical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Summary of categorical columns.
    """

    return dataframe.describe(include=["object"]).T

# ============================================================================
# Duplicate Records
# ============================================================================

def get_duplicate_records(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return all duplicate rows, including their first occurrence.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        All duplicate records.
    """

    return dataframe[
        dataframe.duplicated(keep=False)
    ].sort_index()

# ============================================================================
# Dataset Validation
# ============================================================================

def validate_dataset(dataframe: pd.DataFrame) -> dict:
    """
    Validate the dataset before preprocessing.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Validation report.
    """

    validation_report = {
        "Dataset Empty": bool(dataframe.empty),
        "Contains Missing Values": bool(dataframe.isnull().values.any()),
        "Contains Duplicates": bool(dataframe.duplicated().any()),
        "Total Missing Values": int(dataframe.isnull().sum().sum()),
        "Total Duplicate Rows": int(dataframe.duplicated().sum())
    }

    validation_report["Validation Status"] = (
        "Passed"
        if not validation_report["Dataset Empty"]
        else "Failed"
    )

    return validation_report

# ============================================================================
# Duplicate Handling
# ============================================================================

def remove_duplicate_records(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove duplicate records from the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset after removing duplicate rows.
    """

    cleaned_dataframe = (
        dataframe
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )

    return cleaned_dataframe

# ============================================================================
# Missing Value Analysis
# ============================================================================

def analyze_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Missing value report.
    """

    report = pd.DataFrame({
        "Missing Values": dataframe.isnull().sum(),
        "Missing Percentage": (
            dataframe.isnull().mean() * 100
        ).round(2),
        "Data Type": dataframe.dtypes
    })

    return report[report["Missing Values"] > 0].sort_values(
        by="Missing Values",
        ascending=False
    )

# ============================================================================
# Missing Value Handling
# ============================================================================

def fill_missing_values(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing values using the median for numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with missing values handled.
    """

    cleaned_dataframe = dataframe.copy()

    numerical_columns = cleaned_dataframe.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:
        if cleaned_dataframe[column].isnull().sum() > 0:
            cleaned_dataframe[column] = cleaned_dataframe[column].fillna(
                cleaned_dataframe[column].median()
            )

    return cleaned_dataframe

# ============================================================================
# Data Type Validation
# ============================================================================

def validate_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data types of all columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Column names and data types.
    """

    return pd.DataFrame({
        "Column": dataframe.columns,
        "Data Type": dataframe.dtypes.astype(str)
    })

# ============================================================================
# Dataset Export
# ============================================================================

def save_cleaned_dataset(
    dataframe: pd.DataFrame,
    file_path: Path
) -> None:
    """
    Save cleaned dataset to CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset.

    file_path : Path
        Output CSV path.
    """

    dataframe.to_csv(
        file_path,
        index=False
    )