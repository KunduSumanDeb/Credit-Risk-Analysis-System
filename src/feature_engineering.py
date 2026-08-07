# Feature engineering functions
# =============================================================================
# Future Imports
# =============================================================================

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================

# (Nothing required yet)

# =============================================================================
# Third-Party Imports
# =============================================================================

import pickle

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)

# =============================================================================
# Local Application Imports
# =============================================================================

# Will be added later
# from src.config import *
# from src.constants import *

def _validate_dataframe(
    dataframe: pd.DataFrame
) -> None:
    """
    Validate that the input is a non-empty pandas DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to validate.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.

    ValueError
        If the DataFrame is empty.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected 'dataframe' to be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "The input DataFrame is empty."
        )

def _validate_columns_exist(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Validate that the required columns exist in a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to validate.

    columns : list[str]
        List of required column names.

    Raises
    ------
    TypeError
        If columns is not a list.
        If any column name is not a string.

    ValueError
        If one or more required columns are missing.
    """
    # -------------------------------------------------------------------------
    # Validate DataFrame
    # -------------------------------------------------------------------------
    _validate_dataframe(dataframe)

    # -------------------------------------------------------------------------
    # Validate columns parameter
    # -------------------------------------------------------------------------
    if not isinstance(columns, list):
        raise TypeError(
            "Expected 'columns' to be a list of column names."
        )

    if not all(isinstance(column, str) for column in columns):
        raise TypeError(
            "All values in 'columns' must be strings."
        )

    # -------------------------------------------------------------------------
    # Check for missing columns
    # -------------------------------------------------------------------------
    missing_columns = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing "
            f"from the DataFrame: {missing_columns}"
        )

# ============================================================================
# Public Functions
# ============================================================================

def get_numerical_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    excluded_columns: list[str] | None = None,
) -> list[str]:
    """
    Return the numerical feature columns from a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.

    target_column : str
        Name of the target column to exclude.

    excluded_columns : list[str] | None, optional
        Additional columns to exclude.
        Default is None.

    Returns
    -------
    list[str]
        List of numerical feature column names.

    Raises
    ------
    TypeError
        If target_column is not a string.
        If excluded_columns is not a list or None.

    ValueError
        If the target column does not exist.
    """
    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    _validate_dataframe(dataframe)

    if not isinstance(target_column, str):
        raise TypeError(
            "Expected 'target_column' to be a string."
        )
    
    _validate_columns_exist(dataframe, [target_column])

    if excluded_columns is None:
        excluded_columns = []

    if not isinstance(excluded_columns, list):
        raise TypeError(
            "Expected 'excluded_columns' to be a list or None."
        )

    # -------------------------------------------------------------------------
    # Build exclusion set
    # -------------------------------------------------------------------------
    excluded = set(excluded_columns)
    excluded.add(target_column)

    # -------------------------------------------------------------------------
    # Identify numerical columns
    # -------------------------------------------------------------------------
    numerical_columns = [
        column
        for column in dataframe.select_dtypes(include=["number"]).columns
        if column not in excluded
    ]

    return numerical_columns

def get_categorical_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    excluded_columns: list[str] | None = None,
) -> list[str]:
    """
    Return the categorical feature columns from a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.

    target_column : str
        Name of the target column to exclude.

    excluded_columns : list[str] | None, optional
        Additional columns to exclude.
        Default is None.

    Returns
    -------
    list[str]
        List of categorical feature column names.

    Raises
    ------
    TypeError
        If target_column is not a string.
        If excluded_columns is not a list or contains non-string values.

    ValueError
        If the target column does not exist.
    """
    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    _validate_dataframe(dataframe)

    if not isinstance(target_column, str):
        raise TypeError(
            "Expected 'target_column' to be a string."
        )

    _validate_columns_exist(dataframe, [target_column])

    if excluded_columns is None:
        excluded_columns = []

    if not isinstance(excluded_columns, list):
        raise TypeError(
            "Expected 'excluded_columns' to be a list or None."
        )

    if not all(isinstance(column, str) for column in excluded_columns):
        raise TypeError(
            "All values in 'excluded_columns' must be strings."
        )

    # -------------------------------------------------------------------------
    # Build exclusion set
    # -------------------------------------------------------------------------
    excluded = set(excluded_columns)
    excluded.add(target_column)

    # -------------------------------------------------------------------------
    # Identify categorical columns
    # -------------------------------------------------------------------------
    categorical_columns = [
        column
        for column in dataframe.select_dtypes(
            include=["object", "category"]
        ).columns
        if column not in excluded
    ]

    return categorical_columns

def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into feature matrix and target vector.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing features and target.

    target_column : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing:

        - X : pd.DataFrame
            Feature matrix.

        - y : pd.Series
            Target vector.

    Raises
    ------
    TypeError
        If target_column is not a string.

    ValueError
        If the target column does not exist.
    """
    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    _validate_dataframe(dataframe)

    if not isinstance(target_column, str):
        raise TypeError(
            "Expected 'target_column' to be a string."
        )

    _validate_columns_exist(
        dataframe,
        [target_column]
    )

    # -------------------------------------------------------------------------
    # Split features and target
    # -------------------------------------------------------------------------
    X = dataframe.drop(columns=[target_column])

    y = dataframe[target_column]

    return X, y

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset into training, validation, and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.

    test_size : float, optional
        Proportion of the dataset reserved for testing.
        Default is TEST_SIZE.

    validation_size : float, optional
        Proportion of the remaining training data reserved
        for validation.
        Default is VALIDATION_SIZE.

    random_state : int, optional
        Random seed for reproducibility.
        Default is RANDOM_STATE.

    stratify : bool, optional
        Whether to perform stratified splitting.
        Default is True.

    Returns
    -------
    tuple
        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
        )

    Raises
    ------
    TypeError
        If the input types are invalid.

    ValueError
        If the inputs contain invalid values.
    """
    # -------------------------------------------------------------------------
    # Validate feature matrix
    # -------------------------------------------------------------------------
    _validate_dataframe(X)

    # -------------------------------------------------------------------------
    # Validate target vector
    # -------------------------------------------------------------------------
    if not isinstance(y, pd.Series):
        raise TypeError(
            "Expected 'y' to be a pandas Series."
        )

    if y.empty:
        raise ValueError(
            "The target Series is empty."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    # -------------------------------------------------------------------------
    # Validate split parameters
    # -------------------------------------------------------------------------
    if not isinstance(test_size, float):
        raise TypeError(
            "Expected 'test_size' to be a float."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "'test_size' must be between 0 and 1."
        )

    if not isinstance(validation_size, float):
        raise TypeError(
            "Expected 'validation_size' to be a float."
        )

    if not 0 < validation_size < 1:
        raise ValueError(
            "'validation_size' must be between 0 and 1."
        )

    if not isinstance(random_state, int):
        raise TypeError(
            "Expected 'random_state' to be an integer."
        )

    if not isinstance(stratify, bool):
        raise TypeError(
            "Expected 'stratify' to be a boolean."
        )

    # -------------------------------------------------------------------------
    # First Split : Train + Validation / Test
    # -------------------------------------------------------------------------
    stratify_target = y if stratify else None

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # -------------------------------------------------------------------------
    # Second Split : Train / Validation
    # -------------------------------------------------------------------------
    stratify_target = y_train if stratify else None

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # -------------------------------------------------------------------------
    # Return split datasets
    # -------------------------------------------------------------------------
    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numerical and categorical features.

    Parameters
    ----------
    numerical_columns : list[str]
        List of numerical feature column names.

    categorical_columns : list[str]
        List of categorical feature column names.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline.

    Raises
    ------
    TypeError
        If the inputs are not lists.

    ValueError
        If both column lists are empty.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------

    if not isinstance(numerical_columns, list):
        raise TypeError(
            "Expected 'numerical_columns' to be a list."
        )

    if not isinstance(categorical_columns, list):
        raise TypeError(
            "Expected 'categorical_columns' to be a list."
        )

    if not all(
        isinstance(column, str)
        for column in numerical_columns
    ):
        raise TypeError(
            "All numerical column names must be strings."
        )

    if not all(
        isinstance(column, str)
        for column in categorical_columns
    ):
        raise TypeError(
            "All categorical column names must be strings."
        )

    if (
        len(numerical_columns) == 0
        and
        len(categorical_columns) == 0
    ):
        raise ValueError(
            "At least one feature column must be provided."
        )

    # -------------------------------------------------------------------------
    # Numerical Pipeline
    # -------------------------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Categorical Pipeline
    # -------------------------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Build Preprocessor
    # -------------------------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor

def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Fit a preprocessing pipeline using the training dataset.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline returned by
        build_preprocessor().

    X_train : pd.DataFrame
        Training feature matrix.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessing pipeline.

    Raises
    ------
    TypeError
        If the inputs are of incorrect type.

    ValueError
        If the training DataFrame is empty.
    """

    # -------------------------------------------------------------------------
    # Validate preprocessor
    # -------------------------------------------------------------------------

    if not isinstance(
        preprocessor,
        ColumnTransformer,
    ):
        raise TypeError(
            "Expected 'preprocessor' to be a ColumnTransformer."
        )

    # -------------------------------------------------------------------------
    # Validate training data
    # -------------------------------------------------------------------------

    _validate_dataframe(X_train)

    # -------------------------------------------------------------------------
    # Fit preprocessing pipeline
    # -------------------------------------------------------------------------

    preprocessor.fit(X_train)

    return preprocessor

def transform_dataset(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Transform the training, validation and test datasets using
    a fitted preprocessing pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A fitted preprocessing pipeline.

    X_train : pd.DataFrame
        Training feature matrix.

    X_validation : pd.DataFrame
        Validation feature matrix.

    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Transformed training,
        validation,
        and test feature matrices.

    Raises
    ------
    TypeError
        If the inputs are of incorrect type.
    """

    # -------------------------------------------------------------------------
    # Validate preprocessor
    # -------------------------------------------------------------------------

    if not isinstance(
        preprocessor,
        ColumnTransformer,
    ):
        raise TypeError(
            "Expected 'preprocessor' to be a ColumnTransformer."
        )

    # -------------------------------------------------------------------------
    # Validate datasets
    # -------------------------------------------------------------------------

    _validate_dataframe(X_train)
    _validate_dataframe(X_validation)
    _validate_dataframe(X_test)

    # -------------------------------------------------------------------------
    # Transform datasets
    # -------------------------------------------------------------------------

    X_train_transformed = preprocessor.transform(X_train)

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    # -------------------------------------------------------------------------
    # Retrieve feature names
    # -------------------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    # -------------------------------------------------------------------------
    # Convert to DataFrames
    # -------------------------------------------------------------------------

    X_train_processed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index,
    )

    X_validation_processed = pd.DataFrame(
        X_validation_transformed,
        columns=feature_names,
        index=X_validation.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        X_train_processed,
        X_validation_processed,
        X_test_processed,
    )

def save_pickle(
    object_to_save: object,
    file_path: Path,
) -> None:
    """
    Save a Python object as a pickle file.

    Parameters
    ----------
    object_to_save : object
        Python object to save.

    file_path : pathlib.Path
        Destination file path.

    Raises
    ------
    TypeError
        If file_path is not a Path object.

    FileNotFoundError
        If the parent directory does not exist.
    """

    # -------------------------------------------------------------------------
    # Validate file path
    # -------------------------------------------------------------------------

    if not isinstance(file_path, Path):
        raise TypeError(
            "Expected 'file_path' to be a pathlib.Path object."
        )

    if not file_path.parent.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {file_path.parent}"
        )

    # -------------------------------------------------------------------------
    # Save object
    # -------------------------------------------------------------------------

    with open(
        file_path,
        "wb",
    ) as file:

        pickle.dump(
            object_to_save,
            file,
        )

def load_pickle(
    file_path: Path,
) -> object:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    file_path : pathlib.Path
        Pickle file path.

    Returns
    -------
    object
        Loaded Python object.

    Raises
    ------
    TypeError
        If file_path is not a Path object.

    FileNotFoundError
        If the pickle file does not exist.
    """

    # -------------------------------------------------------------------------
    # Validate file path
    # -------------------------------------------------------------------------

    if not isinstance(file_path, Path):
        raise TypeError(
            "Expected 'file_path' to be a pathlib.Path object."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    # -------------------------------------------------------------------------
    # Load object
    # -------------------------------------------------------------------------

    with open(
        file_path,
        "rb",
    ) as file:

        loaded_object = pickle.load(
            file,
        )

    return loaded_object