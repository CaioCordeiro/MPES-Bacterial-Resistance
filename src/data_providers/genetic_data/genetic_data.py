"""Module for running machine learning experiments on genetic data."""

from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

import constants.constants as const
from data_providers.genetic_data.genetic_data import GeneticDataset
from entities.interfaces.feature_selection import FeatureSelectionInterface


class GeneticDataRun:
    """
    Encapsulates the process of running a machine learning model on genetic data,
    including feature selection and cross-validation.
    """

    def __init__(
        self,
        data: GeneticDataset,
        target: str,
        bac: str,
        model: BaseEstimator,
        fs: FeatureSelectionInterface,
    ) -> None:
        """
        Initializes the GeneticDataRun object.

        Args:
            data: The genetic dataset to be used.
            target: The name of the target variable (e.g., antibiotic resistance).
            bac: The name of the bacteria being analyzed.
            model: The machine learning model to be trained and evaluated.
            fs: The feature selection method to be used.
        """
        self.data: GeneticDataset = data
        self.target: str = target
        self.bac: str = bac
        self.model: BaseEstimator = model
        self.fs: FeatureSelectionInterface = fs

    @property
    def _df(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing features and the target variable,
        excluding other antibiotic resistance columns.

        Returns:
            A Pandas DataFrame with relevant features and the target variable.
        """
        anti_list_without_target: List[str] = const.ANTIBIOTIC_LIST.copy()
        print(const.ANTIBIOTIC_LIST)
        anti_list_without_target.remove(self.target)
        return self.data.treated_data.drop(
            anti_list_without_target, axis=1, errors="ignore"
        )

    @property
    def _X(self) -> pd.DataFrame:
        """
        Returns the feature matrix (independent variables).

        Returns:
            A Pandas DataFrame containing the features.
        """
        return self._df.drop(self.target, axis=1, errors="ignore")

    @property
    def _y(self) -> pd.Series:
        """
        Returns the target variable (dependent variable).

        Returns:
            A Pandas Series containing the target variable.
        """
        return self._df[self.target]

    @property
    def _selected_features(self) -> pd.DataFrame:
        """
        Performs feature selection and returns the DataFrame with selected features.

        Returns:
            A Pandas DataFrame containing the selected features.
        """
        print(f"Starting Feature Selection: {self.fs.name} on bac {self.bac}")
        return self.fs.fit(self._df, self._X, self._y)

    def run(self) -> np.ndarray:
        """
        Runs the machine learning pipeline: performs feature selection and
        evaluates the model using cross-validation.

        Returns:
            A NumPy array containing the cross-validation scores.
        """
        selected_df: pd.DataFrame = self._selected_features
        scores: np.ndarray = cross_val_score(self.model, selected_df, self._y, cv=10)
        return scores
