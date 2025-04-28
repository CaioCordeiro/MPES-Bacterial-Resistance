# /home/ccmr/workspace/MPES-Bacterial-Resistance/src/entities/models/linear_regression_model.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils.logging_config import get_logger


class LinearRegressionModel:
    """
    A wrapper class for various linear regression models with hyperparameter tuning.

    Args:
        model_type (str): The type of linear regression model to use. Can be 'ridge', 'lasso', 'elastic_net', or 'linear'.
        param_grid (Dict[str, List]): A dictionary of hyperparameters and their corresponding values to be tuned.
        cv (int): The number of cross-validation folds to use.
    """

    def __init__(
        self,
        model_type: str = "linear",
        param_grid: Dict[str, List] = None,
        cv: int = 5,
    ):
        self.model_type = model_type
        # Default param_grid for relevant models if None is provided
        if param_grid is None and model_type in ["ridge", "lasso", "elastic_net"]:
            self._param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1, 10],
            }
            # Add l1_ratio for elastic_net if not provided
            if model_type == "elastic_net":
                self._param_grid.setdefault("l1_ratio", [0.1, 0.5, 0.9])
        elif param_grid is not None:
            self._param_grid = param_grid
        else:  # model_type is 'linear' or param_grid is explicitly None
            self._param_grid = {}  # No parameters to tune for LinearRegression

        self.cv = cv
        self.logger = get_logger()
        self.model = None
        self.best_model = None
        self.best_params_ = {}  # Initialize as empty dict
        self.feature_importances_ = None
        self._summary = {}
        self.scaler = StandardScaler()  # Initialize scaler here
        self.X_columns = None  # Initialize X_columns

        # Adjust CV parameters based on available memory
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 70:  # If memory usage is high
            self.logger.warning(
                f"High memory usage ({memory_percent}%). Using memory-efficient settings."
            )
            self.cv = 3  # Reduce cross-validation folds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        """
        Fits the linear regression model to the provided data.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            LinearRegressionModel: The fitted model instance.
        """
        self.X_columns = X.columns.tolist()

        # Check memory before proceeding
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:  # More than 80% memory used
            self.logger.warning(
                f"High memory usage ({memory_percent}%). Using simplified model fitting."
            )
            # Use simplified parameter grid for high memory usage
            if self._param_grid:  # Check if param_grid is not empty
                self._param_grid = {
                    param: grid[:2] for param, grid in self._param_grid.items() if grid
                }  # Ensure grid is not empty

        # Standardize features
        # self.scaler = StandardScaler() # Moved scaler initialization to __init__
        X_scaled = self.scaler.fit_transform(X.values)

        if self.model_type == "linear":
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            self.best_params_ = {}  # No hyperparameters tuned
        elif self.model_type in ["ridge", "lasso", "elastic_net"]:
            # Use GridSearchCV for hyperparameter tuning with memory-efficient settings
            grid_search = GridSearchCV(
                estimator=self._get_model_instance(),
                param_grid=self._param_grid,
                scoring="neg_mean_squared_error",
                cv=self.cv,
                return_train_score=True,
                n_jobs=2,  # Limit to 2 cores to reduce memory usage
                verbose=0,  # Reduce verbosity to minimize output
            )
            grid_search.fit(X_scaled, y)

            self.best_model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_  # Store the best parameters
            self.model = self.best_model
        else:
            raise ValueError(
                f"Invalid model type: {self.model_type}. Supported types are 'linear', 'ridge', 'lasso', 'elastic_net'."
            )

        self._update_summary(X, y)  # Pass original X for scaling within summary update
        return self

    def _get_model_instance(self) -> Union[Ridge, Lasso, ElasticNet]:
        """
        Returns an instance of the specified linear regression model.
        """
        if self.model_type == "ridge":
            return Ridge()
        elif self.model_type == "lasso":
            return Lasso()
        elif self.model_type == "elastic_net":
            return ElasticNet()
        else:
            # This case should ideally be caught before calling this method,
            # but added for robustness.
            raise ValueError(f"Cannot get instance for model type: {self.model_type}")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        """
        Performs cross-validation on the model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
            cv (int): The number of cross-validation folds.

        Returns:
            numpy.ndarray: Cross-validation scores (neg_mean_squared_error)
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before cross_validate()."
            )
        if self.X_columns is None:
            raise RuntimeError(
                "Model has not been fitted with column information. Call fit() first."
            )

        # Check memory before cross-validation
        memory_percent = psutil.virtual_memory().percent
        effective_cv = cv
        if memory_percent > 80:  # More than 80% memory used
            self.logger.warning(
                f"High memory usage ({memory_percent}%). Using reduced cross-validation."
            )
            # Use fewer folds for cross-validation when memory is constrained
            effective_cv = min(cv, 3)

        # Ensure X has the same columns as during fitting, in the same order
        X_aligned = X[self.X_columns]
        X_scaled = self.scaler.transform(X_aligned.values)

        return cross_val_score(
            self.model,
            X_scaled,
            y,
            cv=effective_cv,
            scoring="neg_mean_squared_error",
            n_jobs=2,
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the model's performance and configuration.
        """
        if self.model is None:
            self.logger.warning("Attempted to get summary before model was fitted.")
            return {
                "model_type": self.model_type,
                "status": "Not Fitted",
                "best_params": self.best_params_,  # Return params even if not fitted fully
            }
        return self._summary

    def _update_summary(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Updates the model's performance summary after fitting.
        """
        if self.model is None or self.X_columns is None:
            self.logger.warning(
                "Cannot update summary, model not fitted or columns not set."
            )
            return

        # Ensure X has the same columns as during fitting, in the same order
        X_aligned = X[self.X_columns]
        X_scaled = self.scaler.transform(X_aligned.values)

        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        self._summary = {
            "model_type": self.model_type,
            "best_params": self.best_params_,
            "mean_squared_error": mse,
            "r2_score": r2,
            "n_features": len(self.X_columns),
            "status": "Fitted",
        }

        if hasattr(self.model, "coef_"):
            self.feature_importances_ = self.model.coef_
            # Optionally add coefficients to summary if needed, could be large
            # self._summary['coefficients'] = dict(zip(self.X_columns, self.feature_importances_))
        if hasattr(self.model, "intercept_"):
            self._summary["intercept"] = self.model.intercept_

    def get_best_params(self) -> Dict[str, Any]:
        """
        Returns the best hyperparameters found during GridSearchCV, if applicable.

        Returns:
            Dict[str, Any]: A dictionary containing the best parameter settings.
                            Returns an empty dictionary if hyperparameter tuning
                            was not performed (e.g., for 'linear' model_type or
                            if fit() hasn't been called).
        """
        return self.best_params_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted model.

        Args:
            X (pd.DataFrame): The feature data for which to make predictions.

        Returns:
            np.ndarray: The predicted values.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )
        if self.X_columns is None:
            raise RuntimeError(
                "Model has not been fitted with column information. Call fit() first."
            )

        # Ensure X has the same columns as during fitting, in the same order
        try:
            X_aligned = X[self.X_columns]
        except KeyError as e:
            missing_cols = set(self.X_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.X_columns)
            raise ValueError(
                f"Input columns mismatch. Missing: {missing_cols}, Extra: {extra_cols}"
            ) from e

        X_scaled = self.scaler.transform(X_aligned.values)
        return self.model.predict(X_scaled)

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Returns the feature importances (coefficients) of the linear model.

        Returns:
            Optional[Dict[str, float]]: A dictionary mapping feature names to their
                                        coefficients, or None if not available or
                                        model not fitted.
        """
        if self.feature_importances_ is not None and self.X_columns is not None:
            return dict(zip(self.X_columns, self.feature_importances_))
        else:
            self.logger.warning(
                "Feature importances not available. Model might not be fitted or doesn't have coefficients."
            )
            return None
