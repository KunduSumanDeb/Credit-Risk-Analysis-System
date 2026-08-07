# Visualization utilities
"""
Visualization Module
====================

This module contains reusable visualization functions
for Exploratory Data Analysis (EDA).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================================
# Plot Configuration
# ============================================================================

plt.style.use("ggplot")

DEFAULT_FIGURE_SIZE = (10, 6)

TITLE_SIZE = 16

LABEL_SIZE = 12

def load_cleaned_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset.

    Parameters
    ----------
    file_path : Path
        Path to cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded cleaned dataset.
    """

    dataframe = pd.read_csv(file_path)

    return dataframe

def get_dataset_overview(
    dataframe: pd.DataFrame
) -> dict:
    """
    Generate a basic overview of the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Dataset overview.
    """

    overview = {
        "Rows": int(dataframe.shape[0]),
        "Columns": int(dataframe.shape[1]),
        "Numerical Columns": int(
            dataframe.select_dtypes(include="number").shape[1]
        ),
        "Categorical Columns": int(
            dataframe.select_dtypes(exclude="number").shape[1]
        ),
        "Memory Usage (MB)": float(
            round(
                dataframe.memory_usage(deep=True).sum()
                / (1024 ** 2),
                2
            )
        )
    }

    return overview

# ============================================================================
# Target Distribution Summary
# ============================================================================

def get_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Generate count and percentage of the target variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    target_column : str
        Target column.

    Returns
    -------
    pd.DataFrame
        Count and percentage of each target class.
    """

    summary = (
        dataframe[target_column]
        .value_counts()
        .rename("Count")
        .to_frame()
    )

    summary["Percentage"] = (
        summary["Count"] /
        summary["Count"].sum() * 100
    ).round(2)

    summary.index.name = target_column

    return summary

# ============================================================================
# Target Variable Distribution
# ============================================================================

def plot_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str
):
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    target_column : str
        Target column.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object.
    """

    figure, axis = plt.subplots(
        figsize=DEFAULT_FIGURE_SIZE
    )

    sns.countplot(
        data=dataframe,
        x=target_column,
        hue=target_column,
        palette="Set2",
        legend=False,
        ax=axis
    )

    total = len(dataframe)

    for patch in axis.patches:

        height = patch.get_height()

        percentage = height / total * 100

        axis.annotate(
            f"{int(height)}\n({percentage:.1f}%)",
            (
                patch.get_x() + patch.get_width()/2,
                height
            ),
            ha="center",
            va="bottom",
            fontsize=10
        )

    axis.set_title(
        "Loan Status Distribution",
        fontsize=TITLE_SIZE
    )

    axis.set_xlabel("Loan Status")

    axis.set_ylabel("Number of Customers")

    plt.tight_layout()

    return axis

# ============================================================================
# Numerical Feature Distribution
# ============================================================================

def plot_numerical_distributions(
    dataframe: pd.DataFrame
):
    """
    Plot histograms for all numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    None
    """

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    dataframe[numerical_columns].hist(
        figsize=(18, 12),
        bins=30,
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Numerical Features",
        fontsize=18
    )

    plt.tight_layout()

    return

# ============================================================================
# Numerical Summary
# ============================================================================

def get_numerical_summary(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics.
    """

    summary = dataframe.describe().T

    summary["Skewness"] = (
        dataframe
        .select_dtypes(include="number")
        .skew()
    )

    return summary.round(2)

# ============================================================================
# Numerical Feature Boxplots
# ============================================================================

def plot_boxplots(
    dataframe: pd.DataFrame
):
    """
    Plot boxplots for all numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the boxplots.
    """

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    number_of_columns = len(numerical_columns)

    figure, axes = plt.subplots(
        nrows=(number_of_columns + 1) // 2,
        ncols=2,
        figsize=(15, 4 * ((number_of_columns + 1) // 2))
    )

    axes = axes.flatten()

    for index, column in enumerate(numerical_columns):

        sns.boxplot(
            x=dataframe[column],
            ax=axes[index],
            color="skyblue"
        )

        axes[index].set_title(column)

    for index in range(
        len(numerical_columns),
        len(axes)
    ):
        figure.delaxes(axes[index])

    plt.tight_layout()

    return figure

# ============================================================================
# Correlation Heatmap
# ============================================================================

def plot_correlation_heatmap(
    dataframe: pd.DataFrame
):
    """
    Plot the correlation matrix for numerical features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object.
    """

    numerical_dataframe = dataframe.select_dtypes(
        include="number"
    )

    correlation_matrix = numerical_dataframe.corr()

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=axis
    )

    axis.set_title(
        "Correlation Heatmap",
        fontsize=TITLE_SIZE
    )

    plt.tight_layout()

    return axis

# ============================================================================
# Categorical Summary
# ============================================================================

def get_categorical_summary(
    dataframe: pd.DataFrame
) -> dict:
    """
    Generate summary statistics for categorical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Summary for each categorical column.
    """

    summary = {}

    categorical_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns

    for column in categorical_columns:

        summary[column] = dataframe[column].value_counts()

    return summary

# ============================================================================
# Categorical Feature Distribution
# ============================================================================

def plot_categorical_distributions(
    dataframe: pd.DataFrame
):
    """
    Plot count plots for categorical features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    None
    """

    categorical_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns

    number_of_columns = len(categorical_columns)

    figure, axes = plt.subplots(
        nrows=(number_of_columns + 1) // 2,
        ncols=2,
        figsize=(15, 5 * ((number_of_columns + 1) // 2))
    )

    axes = axes.flatten()

    for index, column in enumerate(categorical_columns):

        sns.countplot(
            data=dataframe,
            x=column,
            hue=column,
            legend=False,
            ax=axes[index]
        )

        axes[index].set_title(column)
        axes[index].tick_params(
            axis="x",
            rotation=30
        )

    for index in range(
        len(categorical_columns),
        len(axes)
    ):
        figure.delaxes(axes[index])

    plt.tight_layout()

# ============================================================================
# Feature vs Target
# ============================================================================

def plot_feature_vs_target(
    dataframe: pd.DataFrame,
    feature: str,
    target_column: str
):
    """
    Plot a feature against the target variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    feature : str
        Feature column.

    target_column : str
        Target column.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object.
    """

    figure, axis = plt.subplots(
        figsize=DEFAULT_FIGURE_SIZE
    )

    sns.boxplot(
        data=dataframe,
        x=target_column,
        y=feature,
        ax=axis
    )

    axis.set_title(
        f"{feature} vs {target_column}",
        fontsize=TITLE_SIZE
    )

    plt.tight_layout()

    return axis

# ============================================================================
# IQR Outlier Detection
# ============================================================================

def detect_outliers_iqr(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Detect outliers using the IQR method.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Outlier count and percentage.
    """

    report = []

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = dataframe[
            (dataframe[column] < lower) |
            (dataframe[column] > upper)
        ]

        report.append({
            "Feature": column,
            "Outliers": len(outliers),
            "Percentage": round(
                len(outliers) / len(dataframe) * 100,
                2
            )
        })

    return pd.DataFrame(report)

# ============================================================================
# Save Figure
# ============================================================================

def save_figure(
    figure,
    file_path: Path
):
    """
    Save a matplotlib figure.

    Parameters
    ----------
    figure
        Matplotlib figure object.

    file_path : Path
        Destination path.

    Returns
    -------
    None
    """

    figure.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )