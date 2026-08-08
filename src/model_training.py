# Model training functions
# =============================================================================
# Future Imports
# =============================================================================

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================

from typing import Any

# =============================================================================
# Third-Party Imports
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

# =============================================================================
# Local Application Imports
# =============================================================================

from src.config import RANDOM_STATE

# Local imports will be added as required.
# from src.config import RANDOM_STATE


# =============================================================================
# Validation Helpers
# =============================================================================

def _validate_model(
    model: Any,
) -> None:
    """
    Validate that the provided object is a valid scikit-learn estimator.

    The estimator must inherit from scikit-learn's BaseEstimator class
    and provide both fit() and predict() methods.

    Parameters
    ----------
    model : Any
        Machine learning model to validate.

    Raises
    ------
    TypeError
        If the model is None.

    TypeError
        If the model is not a scikit-learn estimator.

    TypeError
        If the model does not provide a fit() method.

    TypeError
        If the model does not provide a predict() method.
    """

    # -------------------------------------------------------------------------
    # Validate model is not None
    # -------------------------------------------------------------------------

    if model is None:
        raise TypeError(
            "Expected 'model' to be a scikit-learn estimator, "
            "but received None."
        )

    # -------------------------------------------------------------------------
    # Validate scikit-learn estimator
    # -------------------------------------------------------------------------

    if not isinstance(
        model,
        BaseEstimator,
    ):
        raise TypeError(
            "Expected 'model' to be a scikit-learn estimator "
            "inheriting from BaseEstimator."
        )

    # -------------------------------------------------------------------------
    # Validate fit method
    # -------------------------------------------------------------------------

    if not callable(
        getattr(model, "fit", None)
    ):
        raise TypeError(
            "The provided model does not have a callable 'fit()' method."
        )

    # -------------------------------------------------------------------------
    # Validate predict method
    # -------------------------------------------------------------------------

    if not callable(
        getattr(model, "predict", None)
    ):
        raise TypeError(
            "The provided model does not have a callable 'predict()' method."
        )

def build_logistic_regression(
    **kwargs: Any,
) -> LogisticRegression:
    """
    Build and configure a Logistic Regression classifier.

    The function creates an untrained Logistic Regression estimator
    using the project's configured random state and a sufficiently
    large iteration limit for model convergence.

    Additional scikit-learn LogisticRegression parameters can be
    supplied through keyword arguments.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments accepted by
        sklearn.linear_model.LogisticRegression.

        These parameters override the default values supplied by
        this function.

    Returns
    -------
    LogisticRegression
        An untrained Logistic Regression estimator.

    Raises
    ------
    TypeError
        If an invalid parameter type is supplied to the
        LogisticRegression constructor.

    ValueError
        If an invalid parameter value is supplied to the
        LogisticRegression constructor.
    """

    model_parameters = {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    model_parameters.update(kwargs)

    model = LogisticRegression(
        **model_parameters,
    )

    return model

def build_decision_tree(
    **kwargs: Any,
) -> DecisionTreeClassifier:
    """
    Build and configure a Decision Tree classifier.

    The function creates an untrained Decision Tree estimator using
    the project's configured random state. Additional parameters
    supported by sklearn.tree.DecisionTreeClassifier can be supplied
    through keyword arguments.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments accepted by
        sklearn.tree.DecisionTreeClassifier.

        Supplied parameters override the default values defined by
        this function.

    Returns
    -------
    DecisionTreeClassifier
        An untrained Decision Tree classifier.

    Raises
    ------
    TypeError
        If an invalid parameter name or parameter type is supplied
        to the DecisionTreeClassifier constructor.

    ValueError
        If an invalid parameter value is supplied to the
        DecisionTreeClassifier constructor.
    """

    model_parameters = {
        "random_state": RANDOM_STATE,
    }

    model_parameters.update(kwargs)

    model = DecisionTreeClassifier(
        **model_parameters,
    )

    return model

def build_random_forest(
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Build and configure a Random Forest classifier.

    The function creates an untrained Random Forest estimator using
    the project's configured random state. Additional parameters
    supported by sklearn.ensemble.RandomForestClassifier can be
    supplied through keyword arguments.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments accepted by
        sklearn.ensemble.RandomForestClassifier.

        Supplied parameters override the default values defined by
        this function.

    Returns
    -------
    RandomForestClassifier
        An untrained Random Forest classifier.

    Raises
    ------
    TypeError
        If an invalid parameter name or parameter type is supplied
        to the RandomForestClassifier constructor.

    ValueError
        If an invalid parameter value is supplied to the
        RandomForestClassifier constructor.
    """

    model_parameters = {
        "random_state": RANDOM_STATE,
    }

    model_parameters.update(kwargs)

    model = RandomForestClassifier(
        **model_parameters,
    )

    return model

def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Train a machine learning model using the provided training data.

    The function validates the model and training datasets before
    fitting the model. It is model-agnostic and can therefore be
    used with different scikit-learn estimators.

    Parameters
    ----------
    model : BaseEstimator
        Untrained scikit-learn estimator to be fitted.

    X_train : pandas.DataFrame
        Training feature dataset.

    y_train : pandas.Series
        Training target values.

    Returns
    -------
    BaseEstimator
        The fitted machine learning model.

    Raises
    ------
    TypeError
        If the model is not a valid scikit-learn estimator.

    TypeError
        If X_train is not a pandas DataFrame.

    TypeError
        If y_train is not a pandas Series.

    ValueError
        If X_train is empty.

    ValueError
        If y_train is empty.

    ValueError
        If X_train and y_train contain different numbers of
        observations.
    """

    # -------------------------------------------------------------------------
    # Validate model
    # -------------------------------------------------------------------------

    _validate_model(model)

    # -------------------------------------------------------------------------
    # Validate training features
    # -------------------------------------------------------------------------

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "Expected 'X_train' to be a pandas DataFrame."
        )

    if X_train.empty:
        raise ValueError(
            "Training feature dataset 'X_train' cannot be empty."
        )

    # -------------------------------------------------------------------------
    # Validate training target
    # -------------------------------------------------------------------------

    if not isinstance(y_train, pd.Series):
        raise TypeError(
            "Expected 'y_train' to be a pandas Series."
        )

    if y_train.empty:
        raise ValueError(
            "Training target dataset 'y_train' cannot be empty."
        )

    # -------------------------------------------------------------------------
    # Validate feature-target length consistency
    # -------------------------------------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number "
            "of observations."
        )

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    return model

def predict_model(
    model: BaseEstimator,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate class predictions using a fitted machine learning model.

    The function validates the supplied model and feature dataset,
    verifies that the model has already been fitted, and generates
    class predictions using the model's predict method.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn estimator capable of generating
        class predictions.

    X : pandas.DataFrame
        Feature dataset on which predictions will be generated.

    Returns
    -------
    numpy.ndarray
        Predicted class labels generated by the fitted model.

    Raises
    ------
    TypeError
        If the model is not a valid scikit-learn estimator.

    TypeError
        If X is not a pandas DataFrame.

    ValueError
        If X is empty.

    sklearn.exceptions.NotFittedError
        If the supplied model has not been fitted.
    """

    # -------------------------------------------------------------------------
    # Validate model
    # -------------------------------------------------------------------------

    _validate_model(model)

    # -------------------------------------------------------------------------
    # Validate feature dataset
    # -------------------------------------------------------------------------

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Expected 'X' to be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Prediction feature dataset 'X' cannot be empty."
        )

    # -------------------------------------------------------------------------
    # Verify that the model has been fitted
    # -------------------------------------------------------------------------

    check_is_fitted(model)

    # -------------------------------------------------------------------------
    # Generate predictions
    # -------------------------------------------------------------------------

    predictions = model.predict(X)

    return np.asarray(predictions)

def predict_probability(
    model: BaseEstimator,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate class probabilities using a fitted classification model.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn classification estimator that supports
        probability prediction.

    X : pandas.DataFrame
        Feature dataset on which probabilities will be generated.

    Returns
    -------
    numpy.ndarray
        Two-dimensional array containing the predicted probability
        for each class.

    Raises
    ------
    TypeError
        If the model is not a valid scikit-learn estimator.

    TypeError
        If X is not a pandas DataFrame.

    ValueError
        If X is empty.

    AttributeError
        If the model does not support probability prediction.

    sklearn.exceptions.NotFittedError
        If the supplied model has not been fitted.
    """

    # -------------------------------------------------------------------------
    # Validate model
    # -------------------------------------------------------------------------

    _validate_model(model)

    # -------------------------------------------------------------------------
    # Validate feature dataset
    # -------------------------------------------------------------------------

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Expected 'X' to be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Prediction feature dataset 'X' cannot be empty."
        )

    # -------------------------------------------------------------------------
    # Verify that the model has been fitted
    # -------------------------------------------------------------------------

    check_is_fitted(model)

    # -------------------------------------------------------------------------
    # Verify probability prediction support
    # -------------------------------------------------------------------------

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The supplied model does not support probability prediction."
        )

    # -------------------------------------------------------------------------
    # Generate probability predictions
    # -------------------------------------------------------------------------

    probabilities = model.predict_proba(X)

    return np.asarray(probabilities)