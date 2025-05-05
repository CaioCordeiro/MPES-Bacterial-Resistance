from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from utils.logging_config import get_logger


class LogisticRegressionModel:
    """
    A wrapper class for logistic regression with hyperparameter tuning.

    Args:
        param_grid (Dict[str, List]): Hyperparameters and their values to be tuned.
        cv (int): Number of cross-validation folds.
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 5,
        scoring: str = "f1_weighted",
    ):
        self.param_grid = param_grid or {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        }
        self.cv = cv
        self.scoring = scoring
        self.logger = get_logger()
        self.model = None
        self.best_model = None
        self.best_params_ = {}
        self.feature_importances_ = None
        self._summary = {}
        self.scaler = StandardScaler()
        self.X_columns = None
        self.model_type = "logistic_regression"

        # Adjust CV parameters based on available memory
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 70:
            self.logger.warning(
                f"High memory usage ({memory_percent}%). Using memory-efficient settings."
            )
            self.cv = 3

    def _get_model_instance(self) -> LogisticRegression:
        return LogisticRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self.X_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X.values)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        grid_search = GridSearchCV(
            estimator=self._get_model_instance(),
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            return_train_score=True,
            n_jobs=2,
            verbose=0,
        )
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            self.logger.error(f"GridSearchCV failed for LogisticRegression: {e}", exc_info=True)
            self._summary = {"status": "Fit Failed", "error": str(e)}
            self.logger.warning("Attempting to fit with default LogisticRegression parameters as fallback.")
            try:
                self.model = self._get_model_instance()
                self.model.fit(X_train, y_train)
                self.best_params_ = self.model.get_params()
                self.best_model = self.model
                self.logger.info("Fallback LogisticRegression fitting with defaults succeeded.")
            except Exception as fallback_e:
                self.logger.error(
                    f"Fallback LogisticRegression fitting failed: {fallback_e}", exc_info=True
                )
                self._summary["fallback_error"] = str(fallback_e)
                return self
        else:
            self.best_model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            self.model = self.best_model

        if self.model:
            self._update_summary(X_test, y_test)
        return self

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        if self.model is None:
            self.model = self._get_model_instance()
        return cross_val_score(
            self.model,
            X,
            y,
            cv=10,
            scoring=self.scoring,
            n_jobs=2,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")
        if self.X_columns is None:
            raise RuntimeError("Model has not been fitted with column information. Call fit() first.")

        X_aligned = X[self.X_columns]
        X_scaled = self.scaler.transform(X_aligned.values)
        return self.model.predict(X_scaled)

    def get_summary(self) -> Dict[str, Any]:
        if self.model is None:
            self.logger.warning("Attempted to get summary before model was fitted.")
            return {
                "model_type": "logistic_regression",
                "status": "Not Fitted",
                "best_params": self.best_params_,
            }
        return self._summary

    def _update_summary(self, X_test, y_test) -> None:
        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        self._summary = {
            "model_type": "logistic_regression",
            "scoring": self.scoring,
            "best_params": self.best_params_,
            "n_features": len(self.X_columns),
            "status": "Fitted",
            **metrics,
        }
        if hasattr(self.model, "coef_"):
            self.feature_importances_ = self.model.coef_[0]
        if hasattr(self.model, "intercept_"):
            self._summary["intercept"] = self.model.intercept_

    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params_

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        if self.feature_importances_ is not None and self.X_columns is not None:
            return dict(zip(self.X_columns, self.feature_importances_))
        else:
            self.logger.warning(
                "Feature importances not available. Model might not be fitted or doesn't have coefficients."
            )
            return None