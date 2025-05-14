import warnings  # To suppress potential division by zero warnings if ranges are zero

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array, check_X_y, shuffle

from entities.interfaces.feature_selection import FeatureSelectionInterface
from utils.logging_config import get_logger


# --- RReliefF Implementation (Refactored) ---
class RReliefF(FeatureSelectionInterface):
    """
    Implementation of the RReliefF (Regression ReliefF) feature selection algorithm.

    This version combines score calculation and feature selection (if requested)
    within the `fit` method.

    Attributes:
        n_neighbors (int): The number of nearest neighbors (k) to consider.
        n_features_to_select (int, optional): If specified, the `fit` method
                                              will return the data with the top
                                              `n` features selected. If None,
                                              `fit` returns the feature scores.
                                              Defaults to None.
        n_iterations (int, optional): The number of random instances to sample.
                                      If None, uses all instances. Defaults to None.
        random_state (int, optional): Seed for random number generation.
                                      Defaults to None.
        feature_scores_ (pd.Series): Calculated scores after fitting.
        feature_names_in_ (np.ndarray): Names of features seen during fit.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        n_features_to_select: int = None,
        n_iterations: int = None,
        random_state: int = None,
    ) -> None:
        """
        Initializes the RReliefF algorithm.

        Args:
            n_neighbors (int): The number of nearest neighbors (k).
            n_features_to_select (int, optional): Number of top features to select.
                                                  If None, `fit` returns scores.
            n_iterations (int, optional): Number of sampling iterations. Uses all
                                          instances if None.
            random_state (int, optional): Random seed for reproducibility.
        """
        super().__init__("RReliefF")
        self.logger = get_logger()
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        if n_features_to_select is not None and (
            not isinstance(n_features_to_select, int) or n_features_to_select <= 0
        ):
            raise ValueError("n_features_to_select must be a positive integer or None.")

        self.n_neighbors = n_neighbors
        self.n_features_to_select = n_features_to_select
        self.n_iterations = 1000
        self.random_state = random_state
        self.feature_scores_ = None
        self.feature_names_in_ = None
        self._target_range = None
        self._feature_ranges = None
        self._scores_calculated = False

    def _calculate_scores(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        """
        Internal method to calculate RReliefF scores and store them.

        Args:
            X_scaled (np.ndarray): Scaled feature data (samples x features).
            y_scaled (np.ndarray): Scaled target data (samples).
        """
        n_samples, n_features = X_scaled.shape

        # Calculate ranges for normalization (avoid division by zero)
        epsilon = np.finfo(float).eps
        self._feature_ranges = np.ptp(X_scaled, axis=0) + epsilon
        self._target_range = np.ptp(y_scaled) + epsilon

        # Initialize scores
        self.feature_scores_ = np.zeros(n_features)

        # Determine iterations
        m = self.n_iterations if self.n_iterations is not None else n_samples
        if m > n_samples:
            warnings.warn(
                f"n_iterations ({self.n_iterations}) > n_samples ({n_samples}). Using n_samples instead.",
                UserWarning,
            )
            m = n_samples

        # Determine indices to iterate over
        if m < n_samples:
            rng = np.random.default_rng(self.random_state)
            iter_indices = rng.choice(n_samples, m, replace=False)
        else:
            iter_indices = np.arange(n_samples)
            if self.random_state is not None:
                iter_indices = shuffle(
                    iter_indices, random_state=self.random_state
                )  # Use sklearn shuffle for consistency

        # Nearest Neighbors Calculation
        # Use a more memory-efficient approach for large datasets
        if n_samples > 1000:
            # For large datasets, use a subset for nearest neighbors calculation
            subset_size = min(1000, n_samples)
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
            nn_model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors + 1, subset_size), metric="manhattan"
            )
        else:
            nn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors + 1, metric="manhattan"
            )
        nn_model.fit(X_scaled)
        distances, indices = nn_model.kneighbors(X_scaled)
        neighbor_indices = indices[:, 1:]  # Exclude self

        # Weight Update Loop
        for i in iter_indices:
            current_instance_X = X_scaled[i, :]
            current_instance_y = y_scaled[i]
            k_neighbors_idx = neighbor_indices[i, :]

            # Target Differences
            target_diff_neighbors = np.abs(
                current_instance_y - y_scaled[k_neighbors_idx]
            )

            # Feature Differences (normalized)
            feature_diff_neighbors = np.abs(
                current_instance_X - X_scaled[k_neighbors_idx, :]
            )
            norm_feature_diff_neighbors = feature_diff_neighbors / self._feature_ranges

            # Weighting Factor (based on target proximity)
            weight_factor = np.exp(-((target_diff_neighbors / self._target_range) ** 2))

            # Update Term
            update_term = np.sum(
                norm_feature_diff_neighbors * weight_factor[:, np.newaxis], axis=0
            )

            # RReliefF Update: W(A) = W(A) - update_term
            self.feature_scores_ -= update_term

        # Average scores over iterations
        self.feature_scores_ /= m

        # Store as Pandas Series for convenience (used by _select_features)
        self.feature_scores_ = pd.Series(
            self.feature_scores_, index=self.feature_names_in_
        )
        self._scores_calculated = True

    def _select_features(self, data: DataFrame) -> DataFrame:
        """
        Internal method to select features based on calculated scores.

        Assumes `_calculate_scores` has been called and `self.feature_scores_`
        and `self.n_features_to_select` are set.

        Args:
            data (DataFrame): The original input DataFrame.

        Returns:
            DataFrame: DataFrame with only the selected features (and any
                       non-feature columns like the target).

        Raises:
            ValueError: If input data is missing features seen during fit.
        """
        if not self._scores_calculated or self.feature_scores_ is None:
            # This should ideally not happen if called from fit() correctly
            raise RuntimeError("_select_features called before scores were calculated.")
        if self.n_features_to_select is None:
            # This should also not happen based on fit() logic
            raise RuntimeError(
                "_select_features called when n_features_to_select is None."
            )

        n_select = self.n_features_to_select
        if n_select > len(self.feature_names_in_):
            warnings.warn(
                f"n_features_to_select ({n_select}) is greater than the number of features fitted ({len(self.feature_names_in_)}). Selecting all fitted features.",
                UserWarning,
            )
            n_select = len(self.feature_names_in_)

        # Get top N feature names
        top_features = self.feature_scores_.sort_values(ascending=False).tolist()

        # Check if input data has the required columns
        missing_cols = set(top_features) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Input data is missing columns required for selection: {missing_cols}"
            )

        # Identify columns to keep: selected features + any original columns that were not features
        cols_to_keep = top_features + [
            col for col in data.columns if col not in self.feature_names_in_
        ]
        # Ensure unique columns and preserve original order as much as possible
        cols_to_keep_ordered = [col for col in data.columns if col in cols_to_keep]

        return data[cols_to_keep_ordered]

    def fit(self, data: DataFrame, target_column: str) -> DataFrame:
        """
        Fits the RReliefF algorithm, calculates scores, and optionally selects features.

        Args:
            data (DataFrame): Input DataFrame with features and target.
            target_column (str): Name of the target variable column.

        Returns:
            DataFrame:
                - If `n_features_to_select` was set during initialization:
                  Returns the input DataFrame containing only the selected top features
                  (plus the target column and any other non-feature columns).
                - If `n_features_to_select` was None:
                  Returns a DataFrame with features as index and their calculated
                  RReliefF scores as a column named 'Score'.
        """
        # --- Input Validation and Preparation ---
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        if data.isnull().values.any():
            warnings.warn(
                "Input data contains NaN values. RReliefF might produce unexpected results. Consider imputing missing values first.",
                UserWarning,
            )

        # # Check memory before proceeding
        # memory_percent = psutil.virtual_memory().percent
        # if memory_percent > 80:  # More than 80% memory used
        #     self.logger.warning(
        #         f"High memory usage ({memory_percent}%). Using memory-efficient mode for RReliefF."
        #     )
        #     # Use a smaller subset of features if memory is constrained
        #     if len(data.columns) > 1000:
        #         data = data.iloc[:, :1000]  # Use only the first 1000 features

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Use check_X_y for validation and conversion to numpy
        X_np, y_np = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, y_numeric=True
        )
        n_samples, n_features = X_np.shape
        self.feature_names_in_ = X.columns.to_numpy()  # Store feature names

        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"Expected n_neighbors < n_samples, but got n_neighbors={self.n_neighbors} and n_samples={n_samples}"
            )

        # --- Scaling ---
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_np)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y_np.reshape(-1, 1)).ravel()

        # --- Calculate Scores ---
        self._calculate_scores(X_scaled, y_scaled)  # Sets self.feature_scores_

        # --- Select Features or Return Scores ---
        if self.n_features_to_select is not None:
            # User wants feature selection, call the selection helper
            ranked_features = self.feature_scores_.sort_values(
                ascending=False
            ).index.tolist()
            return ranked_features
        else:
            # User wants scores, return the scores DataFrame
            result_df = pd.DataFrame(self.feature_scores_)  # Already a Series
            result_df.columns = ["Score"]
            result_df.index.name = "Feature"
            # Sort by score descending for clarity
            result_df = result_df.sort_values(by="Score", ascending=False)
            ranked_features = self.feature_scores_.sort_values(
                ascending=False
            ).index.tolist()
            return ranked_features
