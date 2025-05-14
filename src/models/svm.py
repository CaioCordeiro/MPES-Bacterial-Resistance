# /home/ccmr/workspace/MPES-Bacterial-Resistance/src/entities/models/svm_model.py
# (Save this in your models directory)

from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (  # Regression metrics for SVR; Classification metrics for SVC
    accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score, roc_auc_score, precision_recall_curve, auc)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR  # Import SVR and SVC

from utils.logging_config import get_logger


class SupportVectorMachineModel:
    """
    A wrapper class for Support Vector Machine models (SVC/SVR) with hyperparameter tuning,
    mirroring the structure of LinearRegressionModel.

    Args:
        model_type (str): The type of SVM model. 'svc' (default) or 'svr'.
        param_grid (Dict[str, List]): Hyperparameters for GridSearchCV. If None, uses defaults.
        cv (int): Number of cross-validation folds.
        scoring (str, optional): Scoring metric for GridSearchCV and cross_validate.
                                 If None, defaults to 'accuracy' for 'svc' and
                                 'neg_mean_squared_error' for 'svr'.
    """

    def __init__(
        self,
        model_type: str = "svc",  # Default to Support Vector Classification
        param_grid: Dict[str, List] = None,
        cv: int =  StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring: Optional[str] = None,  # Allow explicit scoring override
    ):
        if model_type not in ["svc", "svr"]:
            raise ValueError("model_type must be 'svc' or 'svr'")
        self.model_type = model_type

        # Determine default scoring based on model type if not provided
        if scoring is None:
            self.scoring = (
                "accuracy" if model_type == "svc" else "neg_mean_squared_error"
            )
        else:
            self.scoring = scoring

        # Default param_grid for SVC/SVR if None is provided
        if param_grid is None:
            self._param_grid = {
                "C": [0.1, 1, 10],  # Regularization parameter - Reduced for speed
                "kernel": ["linear", "rbf"],  # Common kernels - Reduced for speed
                "gamma": ["scale", "auto"],  # Kernel coefficient for 'rbf'
                # 'degree': [2, 3] # Only for 'poly' kernel, removed for simplicity
            }
            # Add class_weight for SVC if relevant
            if model_type == "svc":
                self._param_grid.setdefault("class_weight", [None, "balanced"])
        else:
            self._param_grid = param_grid

        self.cv = cv
        self.logger = get_logger()
        self.model = None
        self.best_model = None
        self.best_params_ = {}  # Initialize as empty dict
        self.feature_importances_ = None  # Only available for linear kernel
        self._summary = {}
        self.scaler = StandardScaler()  # Initialize scaler here
        self.X_columns = None  # Initialize X_columns
        self._X = None  # Store the original X for confusion matrix
        self._y = None  # Store the original y for confusion matrix

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupportVectorMachineModel":
        """
        Fits the SVM model to the provided data using GridSearchCV.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            SupportVectorMachineModel: The fitted model instance.
        """
        self.X_columns = X.columns.tolist()
        self._X = X  # Store the original X
        self._y = y  # Store the original y
        # Separate in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=self._get_model_instance(),
            param_grid=self._param_grid,
            scoring=self.scoring,  # Use appropriate scoring metric
            cv=self.cv,
            return_train_score=True,  # Set to False if train scores aren't needed
            n_jobs=2,  # Limit cores
            verbose=0,  # Reduce verbosity
            error_score="raise",  # Raise error if a combination fails
        )
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            self.logger.error(f"GridSearchCV failed for SVM: {e}", exc_info=True)
            self._summary = {"status": "Fit Failed", "error": str(e)}
            # Optionally try fitting with default parameters as fallback
            self.logger.warning(
                "Attempting to fit with default SVM parameters as fallback."
            )
            try:
                self.model = self._get_model_instance()
                self.model.fit(X_train, y_train)
                self.best_params_ = self.model.get_params()  # Store default params
                self.best_model = self.model
                self.logger.info("Fallback SVM fitting with defaults succeeded.")
            except Exception as fallback_e:
                self.logger.error(
                    f"Fallback SVM fitting failed: {fallback_e}", exc_info=True
                )
                self._summary["fallback_error"] = str(fallback_e)
                return self  # Return unfit instance
        else:
            # If GridSearchCV succeeded
            self.best_model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_  # Store the best parameters
            self.model = self.best_model  # self.model is the best estimator found

        # Only update summary if model fitting was successful (either gridsearch or fallback)
        if self.model:
            self._update_summary(
                X_test, y_test
            )  # Pass original X for scaling within summary update
        return self

    def _get_model_instance(self) -> Union[SVC, SVR]:
        """
        Returns an instance of the specified SVM model.
        """
        if self.model_type == "svc":
            # probability=True allows predict_proba but slows down training.
            # Add if needed, otherwise keep False for speed.
            return SVC()
        elif self.model_type == "svr":
            # Consider adding cache_size modification for memory if needed
            return SVR()
        else:
            # Should be caught in __init__, but for safety
            raise ValueError(f"Invalid model_type: {self.model_type}")

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Optional[np.ndarray]:
        """
        Performs cross-validation on the *best* fitted model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
            cv (int): The number of cross-validation folds.

        Returns:
            Optional[numpy.ndarray]: Cross-validation scores based on the instance's
                                     scoring metric, or None if model not fitted or CV fails.
        """
        if self.model is None:
            print("New Model Instance")
            self.model = self._get_model_instance()
        try:
            scores = cross_val_score(
                self.model,  # Use the best model found by fit()
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,  # Use the instance's scoring metric
                n_jobs=2,  # Limit parallelism
            )
            return scores
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}", exc_info=True)
            return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the model's performance and configuration.
        """
        if self.model is None:
            # Return summary even if fit failed but summary was populated with error
            if self._summary and "Fit Failed" in self._summary.get("status", ""):
                return self._summary
            self.logger.warning(
                "Attempted to get summary before model was fitted successfully."
            )
            return {
                "model_type": self.model_type,
                "scoring": self.scoring,
                "status": "Not Fitted",
                "best_params": self.best_params_,  # Return params found so far
            }
        return self._summary

    def _update_summary(self, X_test, y_test) -> None:
        """
        Updates the model's performance summary after fitting.
        Calculates metrics relevant to the model_type (SVC or SVR).
        """
        if self.model is None or self.X_columns is None:
            self.logger.warning(
                "Cannot update summary, model not fitted or columns not set."
            )
            return

        # Ensure X has the same columns as during fitting, in the same order
        X_aligned = X_test[self.X_columns]

        try:
            y_pred = self.model.predict(X_aligned)
        except Exception as e:
            self.logger.error(
                f"Prediction failed during summary update: {e}", exc_info=True
            )
            self._summary = {
                "model_type": self.model_type,
                "scoring": self.scoring,
                "best_params": self.best_params_,
                "status": "Summary Failed (Predict Error)",
                "error": str(e),
            }
            return

        # --- Calculate Metrics Based on Model Type ---
        metrics = {}
        try:
            # y_pred_proba = self.model.predict_proba(X_aligned)
            # Ensure y is suitable for classification metrics (e.g., integer labels)
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            # Use average='weighted' for multiclass or if classes are imbalanced
            # Use average='binary' if strictly binary and want score for positive class
            metrics["f1_score"] = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            # Cast all arrays inside confusion matrix to list
            # This is necessary for JSON serialization
            confusion_matrix_result = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = confusion_matrix_result.tolist()
            # metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
            # precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            # metrics["pr_auc"] = auc(recall, precision)

        except Exception as e:
            self.logger.error(f"Failed to calculate SVC metrics: {e}")
            metrics["metrics_error"] = f"SVC metric calculation failed: {e}"

        # --- Base Summary ---
        self._summary = {
            "model_type": self.model_type,
            "scoring": self.scoring,  # Store the scoring used
            "best_params": self.best_params_,
            "n_features": len(self.X_columns),
            "status": "Fitted",
            **metrics,  # Add calculated metrics
        }

    def get_best_params(self) -> Dict[str, Any]:
        """
        Returns the best hyperparameters found during GridSearchCV.
        """
        return self.best_params_

    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Makes predictions using the fitted SVM model.

        Args:
            X (pd.DataFrame): The feature data for which to make predictions.

        Returns:
            Optional[np.ndarray]: The predicted values (class labels for SVC, continuous for SVR),
                                  or None if prediction fails.
        """
        if self.model is None:
            self.logger.error(
                "Model has not been fitted yet. Call fit() before predict()."
            )
            return None
        if self.X_columns is None:
            self.logger.error(
                "Model has not been fitted with column information. Call fit() first."
            )
            return None

        # Ensure X has the same columns as during fitting, in the same order
        try:
            X_aligned = X[self.X_columns]
        except KeyError as e:
            missing_cols = set(self.X_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.X_columns)
            self.logger.error(
                f"Input columns mismatch for predict. Missing: {missing_cols}, Extra: {extra_cols}"
            )
            raise ValueError(
                f"Input columns mismatch. Missing: {missing_cols}, Extra: {extra_cols}"
            ) from e

        try:
            predictions = self.model.predict(X_aligned)
            return predictions
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            return None

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Returns the feature importances (coefficients) of the SVM model
        IF the kernel used is 'linear'. Otherwise, returns None.

        Note: For multi-class SVC, this returns coefficients related to one boundary.

        Returns:
            Optional[Dict[str, float]]: A dictionary mapping feature names to their
                                        coefficients for a linear kernel, or None.
        """
        if self.model is None:
            self.logger.warning("Model not fitted. Cannot get feature importances.")
            return None

        if (
            self.feature_importances_ is not None
            and self.X_columns is not None
            and getattr(self.model, "kernel", None) == "linear"
        ):
            # For SVR, coef_ is shape (1, n_features). For SVC, it's (n_classes-1, n_features).
            # We'll return the first set of coefficients (relevant for SVR or binary/first boundary SVC).
            try:
                # Use abs() if the sign doesn't matter, otherwise keep original coefs
                # importances = np.abs(self.feature_importances_[0])
                importances = self.feature_importances_[0]
                if len(importances) == len(self.X_columns):
                    return dict(zip(self.X_columns, importances))
                else:
                    self.logger.error(
                        f"Mismatch between coefficients length ({len(importances)}) and columns ({len(self.X_columns)}) for linear kernel."
                    )
                    return None
            except IndexError:
                self.logger.error(
                    "Error accessing coefficients array for linear kernel."
                )
                return None
        else:
            self.logger.warning(
                f"Feature importances (coefficients) are only available for kernel='linear'. Current kernel: {getattr(self.model, 'kernel', 'N/A')}."
            )
            return None

