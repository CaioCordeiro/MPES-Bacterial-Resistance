"""--- Pearson Correlation Selector Implementation ---"""

import os
import sys
import warnings
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from entities.interfaces.feature_selection import FeatureSelectionInterface
from utils.logging_config import get_logger


class PearsonCorrelationSelector(FeatureSelectionInterface):
    """
    Feature selection based on Pearson correlation with the target variable.

    Selects features based on the absolute value of their Pearson correlation
    coefficient with the target variable.

    Attributes:
        n_features_to_select (int, optional): If specified, the `fit` method
                                                will return the data with the top
                                                `n` most correlated features selected.
                                                If None, `fit` returns the absolute
                                                correlation scores. Defaults to None.
        feature_scores_ (pd.Series): Absolute Pearson correlation scores after fitting.
        feature_names_in_ (np.ndarray): Names of features seen during fit.
    """

    def __init__(self, n_features_to_select: int = None) -> None:
        """
        Initializes the PearsonCorrelationSelector.

        Args:
            n_features_to_select (int, optional): Number of top features to select
                                                    based on absolute correlation.
                                                    If None, `fit` returns scores.
        """
        super().__init__("PearsonCorrelationSelector")
        self.logger = get_logger()
        if n_features_to_select is not None and (
            not isinstance(n_features_to_select, int) or n_features_to_select <= 0
        ):
            raise ValueError("n_features_to_select must be a positive integer or None.")

        self.n_features_to_select = n_features_to_select
        self.feature_scores_ = None
        self.feature_names_in_ = None
        self._scores_calculated = False

    def fit(self, data: DataFrame, target_column: str) -> List[str]:
        """
        Calculates Pearson correlation scores and optionally selects features.

        Args:
            data (DataFrame): Input DataFrame with features and target.
            target_column (str): Name of the target variable column.

        Returns:
            Listr[str]: List of feature names sorted by their absolute correlation
            with the target variable in descending order.
        """
        # --- Input Validation and Preparation ---
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        if data.isnull().values.any():
            # Pearson correlation in pandas handles pairwise NaNs by default,
            # but it's good practice to warn or handle them explicitly if desired.
            self.logger.warning(
                "Input data contains NaN values. Pearson correlation might be affected or produce NaNs. Consider imputation.",
                UserWarning,
            )
            # Optionally drop rows with NaNs or impute before correlation:
            # data = data.dropna(subset=[target_column] + data.drop(columns=[target_column]).columns.tolist())

        # Calculate correlation scores
        X = data.drop(columns=[target_column])
        y = data[target_column]
        self.feature_names_in_ = X.columns.to_numpy()
        self.logger.warning(f"Running correlation for {len(self.feature_names_in_)}")
        # --- OPTIMIZED CORRELATION CALCULATION ---
        # Calculate absolute correlation of each feature column with the target series
        # Handle potential NaNs resulting from constant columns or other issues
        self.feature_scores_ = X.corrwith(y).abs().fillna(0.0)
        self._scores_calculated = True
        # Sort features by scores in descending order
        ranked_features = self.feature_scores_.sort_values(ascending=False)
        self.logger.warning(
            f"Done running correlation for {len(self.feature_names_in_)}"
        )
        # return the full list of ranked features with only the feature names
        self.ranked_features_ = ranked_features.index.tolist()
        self.logger.info(
            f"Feature ranking completed. Top features: {self.ranked_features_[:10]}"
        )
        return self.ranked_features_


    def _select_features(self, data: DataFrame) -> DataFrame:
        """
        Internal method to select features based on calculated scores.

        Args:
            data (DataFrame): The original input DataFrame.

        Returns:
            DataFrame: DataFrame with only the selected features.
        """
        if not self._scores_calculated or self.feature_scores_ is None:
            raise RuntimeError("_select_features called before scores were calculated.")

        n_select = min(self.n_features_to_select, len(self.feature_names_in_))

        # Get top N feature names
        top_features = (
            self.feature_scores_.sort_values(ascending=False).index[:n_select].tolist()
        )

        # Check if input data has the required columns
        missing_cols = set(top_features) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Input data is missing columns required for selection: {missing_cols}"
            )

        # Return selected features plus any non-feature columns
        cols_to_keep = top_features + [
            col for col in data.columns if col not in self.feature_names_in_
        ]
        return data[
            list(dict.fromkeys(cols_to_keep))
        ]  # Preserve order and ensure uniqueness
