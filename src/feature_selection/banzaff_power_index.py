"""Banzaff Power Index based feature selection"""

import itertools as it
import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score

from entities.interfaces.feature_selection import FeatureSelectionInterface
from utils.logging_config import get_logger

# Filter out the specific RuntimeWarning related to division by zero in np.corrcoef
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BanzhafFeatureSelector(FeatureSelectionInterface):
    """
    Performs feature selection based on a modified Banzhaf power index
    with parallelization.
    """

    def __init__(
        self,
        n_features_to_select: int = 5,
        p_banzhaf: int = 4,
        n_jobs: int = -1,
        bins: int = 5,
    ):
        """
        Initializes the BanzhafFeatureSelector.

        Args:
            n_features_to_select (int): The number of features to select.
            p_banzhaf (int): The parameter 'p' for the Banzhaf power index calculation.
            n_jobs (int): Number of parallel jobs to run (-1 means all CPUs).
            bins (int): The number of bins to use for histogram-based
                        probability distribution estimation.
        """
        super().__init__(name="GTFE")
        self.n_features_to_select = n_features_to_select
        self.logger = get_logger()
        self.p_banzhaf = min(
            p_banzhaf, 3
        )  # Limit p_banzhaf to reduce computational complexity
        self.n_jobs = n_jobs
        self.bins = bins
        self.columns_arr: List[str] = []
        self._scores_calculated = False
        self.features_size: int = 0
        self.selected_features: List[str] = []
        self.feature_flags: List[int] = []
        self.banzhaf_power: List[int] = []
        self.feature_weights: List[float] = []
        self.sum_relevance_redundancy: List[float] = []
        self.df: pd.DataFrame = pd.DataFrame()
        self.feature_scores_ = None
        self.y: pd.Series = pd.Series()

    @staticmethod
    def _safe_log(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes the natural logarithm, handling zero values."""
        if isinstance(x, np.ndarray):
            return np.where(x > 0, np.log2(x), 0)
        return np.log2(x) if x > 0 else 0

    @staticmethod
    def _calculate_entropy(prob_dist: np.ndarray) -> float:
        """Calculates the entropy of a probability distribution in bits."""
        prob_dist_flat = prob_dist.flatten()
        safe_log_values = BanzhafFeatureSelector._safe_log(prob_dist_flat)
        return -np.sum(prob_dist_flat * safe_log_values)

    def _estimate_joint_distribution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimates the joint probability distribution of two variables."""
        hist, _, _ = np.histogram2d(x, y, bins=self.bins, density=True)
        return hist

    @staticmethod
    def _estimate_conditional_distribution(joint_dist: np.ndarray) -> np.ndarray:
        """Estimates the conditional probability distribution from a joint distribution."""
        total_sum = np.sum(joint_dist)
        return joint_dist / total_sum if total_sum > 0 else np.zeros_like(joint_dist)

    def _conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Computes the conditional mutual information I(X; Y | Z) in bits using histogram-based estimation."""
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        z = np.asarray(z).flatten()

        if not (len(x) == len(y) == len(z)):
            raise ValueError(
                f"Input arrays x ({len(x)}), y ({len(y)}), and z ({len(z)}) must have the same length."
            )

        pxy = self._estimate_conditional_distribution(
            self._estimate_joint_distribution(x, y)
        )
        pxz = self._estimate_conditional_distribution(
            self._estimate_joint_distribution(x, z)
        )
        pyz = self._estimate_conditional_distribution(
            self._estimate_joint_distribution(y, z)
        )

        Hxy = self._calculate_entropy(pxy)
        Hxz = self._calculate_entropy(pxz)
        Hyz = self._calculate_entropy(pyz)

        cmi_val = Hxz + Hyz - Hxy
        return max(0, cmi_val)

    @staticmethod
    def _calculate_tanimoto_coefficient(s1: np.ndarray, s2: np.ndarray) -> float:
        """Calculates the Tanimoto coefficient (Jaccard index) between two arrays."""
        intersection = np.sum(np.logical_and(s1, s2))
        union = np.sum(np.logical_or(s1, s2))
        return intersection / union if union > 0 else 0

    def _calculate_pairwise_tanimoto(self, i: int) -> float:
        """Calculates the average Tanimoto coefficient for a given feature."""
        redundancy_sum = 0
        feature_i = self.df[self.columns_arr[i]].values
        for j in range(self.features_size):
            if i != j:
                feature_j = self.df[self.columns_arr[j]].values
                tanimoto_coeff = self._calculate_tanimoto_coefficient(
                    feature_i, feature_j
                )
                redundancy_sum += tanimoto_coeff
        avg_redundancy = (
            redundancy_sum / (self.features_size - 1) if self.features_size > 1 else 0
        )
        return avg_redundancy

    def _calculate_coalition_power(self, i: int) -> Tuple[int, int]:
        """Calculates the Banzhaf power increment for a given unselected feature."""
        feature_to_evaluate_col = self.columns_arr[i]
        coalition_power = 0
        feature_to_evaluate = self.df[feature_to_evaluate_col].values
        y_values = self.y.values

        for r in range(1, self.p_banzhaf + 1):
            for coalition_tuple in it.combinations(self.selected_features, r):
                coalition = list(coalition_tuple)
                if feature_to_evaluate_col not in coalition:
                    if coalition:
                        coalition_data = self.df[coalition].values
                        conditioning_var = (
                            coalition_data[:, 0]
                            if coalition_data.ndim > 1 and coalition_data.shape[1] > 0
                            else coalition_data.flatten()
                        )
                        if len(conditioning_var) == len(feature_to_evaluate) and len(
                            conditioning_var
                        ) == len(y_values):
                            cmi_val = self._conditional_mutual_information(
                                feature_to_evaluate, y_values, conditioning_var
                            )
                            mi_val = mutual_info_score(feature_to_evaluate, y_values)
                            if cmi_val > mi_val:
                                coalition_power += 1
                    elif (
                        mutual_info_score(feature_to_evaluate, y_values) > 0
                    ):  # Base case
                        coalition_power += 1

        num_coalitions = sum(
            1
            for _ in it.combinations(self.selected_features, r)
            for r in range(1, self.p_banzhaf + 1)
        )
        return coalition_power, num_coalitions

    def fit(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Performs feature selection based on the modified Banzhaf power index.

        Args:
            df (pd.DataFrame): The full dataframe containing features and target.
            X (pd.DataFrame): The dataframe of features.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: A dataframe containing the selected features.
        """
        self.df = df
        self.y = df[target_column]
        self.X = df.drop(target_column, axis=1)
        self.columns_arr = self.X.columns.tolist()
        self.features_size = len(self.columns_arr)
        self.selected_features = []
        self.feature_flags = [0] * self.features_size
        self.banzhaf_power = [0] * self.features_size
        self.feature_weights = [1.0] * self.features_size
        self.sum_relevance_redundancy = [0.0] * self.features_size

        # Calculate initial relevance and redundancy scores (parallelized Tanimoto)
        relevance_scores: List[float] = [
            (
                # Use a more memory-efficient correlation calculation
                abs(
                    np.corrcoef(
                        # Convert to float32 to reduce memory usage
                        self.df[col].astype(np.float32),
                        self.y.values.astype(np.float32),
                    )[0][1]
                )
                # Handle NaN values that might occur
                if len(np.unique(self.df[col])) > 1
                and not np.isnan(np.corrcoef(self.df[col], self.y.values)[0][1])
                else 0 if len(np.unique(self.df[col])) > 1 else 0
            )
            for col in self.columns_arr
        ]
        avg_redundancies: List[float] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_pairwise_tanimoto)(i)
            # Process in smaller batches to reduce memory pressure
            for i in range(
                min(self.features_size, 1000)
            )  # Limit to first 1000 features
        )

        # Fill remaining values if needed
        avg_redundancies.extend([0.0 for _ in range(max(0, self.features_size - 1000))])
        self.sum_relevance_redundancy = [
            rel + red for rel, red in zip(relevance_scores, avg_redundancies)
        ]

        for _ in range(self.n_features_to_select):
            local_feature_scores: List[float] = [0.0] * self.features_size
            for i in range(self.features_size):
                if self.feature_flags[i] != 1:
                    local_feature_scores[i] = (
                        self.sum_relevance_redundancy[i] * self.feature_weights[i]
                    )

            best_feature_index = np.argmax(local_feature_scores)
            if self.feature_flags[best_feature_index] == 1:
                break

            best_feature = self.columns_arr[best_feature_index]
            self.feature_flags[best_feature_index] = 1
            self.selected_features.append(best_feature)

            # Parallelize Banzhaf power calculation for remaining features
            remaining_feature_indices = [
                i for i in range(self.features_size) if self.feature_flags[i] == 0
            ]
            results: List[Tuple[int, int]] = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_coalition_power)(i)
                for i in remaining_feature_indices
            )

            for idx, (power, num_coalitions) in zip(remaining_feature_indices, results):
                self.banzhaf_power[idx] += power
                if num_coalitions > 0:
                    self.feature_weights[idx] += (
                        self.banzhaf_power[idx] / num_coalitions
                    )

        self.feature_scores_ = pd.Series(
            {
                col: self.sum_relevance_redundancy[i] * self.feature_weights[i]
                for i, col in enumerate(self.columns_arr)
            }
        )
        self._scores_calculated = True

        return self.df[self.selected_features]
