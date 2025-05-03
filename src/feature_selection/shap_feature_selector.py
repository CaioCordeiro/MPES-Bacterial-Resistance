import pandas as pd
import numpy as np
import shap
import constants.constants as const
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from typing import List, Optional
from entities.interfaces.feature_selection import FeatureSelectionInterface


class ShapFeatureSelector(FeatureSelectionInterface):
    """
    Performs feature ranking based on SHAP values.
    """

    def __init__(self, model, train_model_before_shap: bool = True):
        """
        Initializes the ShapFeatureSelector.

        Args:
            model: A machine learning model compatible with SHAP.
            train_model_before_shap: Whether to train the model before SHAP evaluation.
        """
        super().__init__(name="SHAPFeatureSelector")
        self.model = clone(model._get_model_instance())
        self.train_model_before_shap = train_model_before_shap
        self.feature_scores_: Optional[dict] = None
        self.ranked_features_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """
        Ranks features based on SHAP values.

        Args:
            df: Input DataFrame containing features and the target variable.
            target_column: Name of the target column in the DataFrame.

        Returns:
            A ranked list of feature names.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Train the model if required
        if self.train_model_before_shap:
            self.model.fit(X, y)
            fitted_model = self.model
        else:
            try:
                check_is_fitted(self.model)
            except Exception as e:
                raise ValueError(
                    "The provided model is not fitted. Set train_model_before_shap=True to train it within the selector."
                ) from e
            fitted_model = self.model

        def model_predict(data):
            # Ensure the input data has the same feature names as the training data
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=X.columns)
            return fitted_model.predict(data)

        # Calculate SHAP values
        sampled_data = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(model_predict, sampled_data)
        shap_values = explainer(X)

        # Compute mean absolute SHAP values for each feature
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        self.feature_scores_ = dict(zip(X.columns, feature_importance))

        # Rank features by importance
        self.ranked_features_ = sorted(self.feature_scores_, key=self.feature_scores_.get, reverse=True)

        return self.ranked_features_

    def get_feature_scores(self) -> dict:
        """
        Returns the feature importance scores.

        Returns:
            A dictionary of feature importance scores.
        """
        if self.feature_scores_ is None:
            raise ValueError("Feature scores have not been calculated. Call fit() first.")
        return self.feature_scores_