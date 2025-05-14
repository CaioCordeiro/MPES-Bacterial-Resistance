from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming ProteinDataset is in a similar location to GeneticDataset
# Adjust the import path if your ProteinDataset class is located elsewhere
from data_providers.protein_data.protein_data import ProteinDataset
from entities.interfaces.feature_selection import FeatureSelectionInterface
from utils.logging_config import get_logger


class ProteinDataRun:
    """
    Encapsulates a single run of a protein data analysis pipeline.
    Returns results as a dictionary.
    """

    def __init__(
        self,
        data: ProteinDataset,
        target: str,
        bac: str,  # Context identifier (e.g., bacteria name or dataset source)
        model: BaseEstimator,
        fs: Optional[FeatureSelectionInterface] = None,
        run_config: Optional[Dict[str, Any]] = None,
        selected_features: Optional[List[str]] = None,
        n_features_to_select: Optional[int] = None,
    ):
        self.logger = get_logger()
        self.data: ProteinDataset = data
        self.target: str = target
        self.bac: str = bac  # Used for context, logging, and file naming
        self.model: BaseEstimator = model
        self.fs: Optional[FeatureSelectionInterface] = fs
        self._selected_features: Optional[List[str]] = (
            selected_features if selected_features else None
        )
        self.n_features_to_select: Optional[int] = (
            n_features_to_select if n_features_to_select else 0
        )

        self.run_config: Dict[str, Any] = run_config if run_config is not None else {}
        # Base config - more details added in run()
        self.run_config.update(
            {
                "database": self.data.name,  # ProteinDataset should have a .name property
                "target": self.target,
                "context": self.bac,  # Using 'context' instead of 'bacteria' if more general
                "model_class": self.model.__class__.__name__,
                "feature_selection": self.fs.name if self.fs else None,
                "selected_features_initial": self._selected_features,  # Store initially passed selected features
                "n_features_requested": (
                    getattr(self.fs, "n_features_to_select", self.n_features_to_select)
                    if self.fs
                    else self.n_features_to_select
                ),
            }
        )

    @property
    def _df(self) -> pd.DataFrame:
        """
        Returns the DataFrame with the target variable and features.
        Assumes data.treated_data is the fully preprocessed DataFrame.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if self.data.treated_data is None or self.data.treated_data.empty:
            self.logger.error("Treated data from ProteinDataset is None or empty.")
            # Return an empty DataFrame with target column if possible to avoid downstream errors,
            # or handle this more gracefully depending on expected behavior.
            return pd.DataFrame(columns=[self.target])

        # Ensure target column exists
        if self.target not in self.data.treated_data.columns:
            self.logger.error(
                f"Target column '{self.target}' not found in treated data columns: {self.data.treated_data.columns.tolist()}"
            )
            raise ValueError(
                f"Target column '{self.target}' not found in treated data."
            )

        return self.data.treated_data.dropna()

    @property
    def _X(self) -> pd.DataFrame:
        """
        Returns the feature matrix (independent variables).

        Returns:
            pd.DataFrame: The feature matrix.
        """
        if self._df.empty:
            return pd.DataFrame()
        return self._df.drop(self.target, axis=1, errors="ignore")

    @property
    def _y(self) -> pd.Series:
        """
        Returns the target vector (dependent variable).

        Returns:
            pd.Series: The target vector.
        """
        if self._df.empty or self.target not in self._df:
            return pd.Series(dtype="float64")  # Return empty series
        return self._df[self.target]

    @property
    def selected_features_df(self) -> pd.DataFrame:
        """
        Performs feature selection if a selector is provided and features aren't pre-selected,
        and returns the DataFrame with selected features.
        If no feature selector, uses all features from _X.
        If _selected_features are provided initially, uses those.

        Returns:
            pd.DataFrame: The DataFrame containing only the selected features.
        """
        if self._X.empty:
            self.logger.warning("_X is empty, cannot select features.")
            return pd.DataFrame()

        if self._selected_features:  # Features were provided in __init__
            self.logger.info(
                f"Using pre-defined selected features: {self._selected_features}"
            )
            # Ensure all pre-selected features are in _X
            missing_features = [
                f for f in self._selected_features if f not in self._X.columns
            ]
            if missing_features:
                self.logger.error(
                    f"Pre-selected features not found in data: {missing_features}"
                )
                raise ValueError(
                    f"Pre-selected features not found in data: {missing_features}"
                )
            return self._X[self._selected_features]

        if self.fs:
            self.logger.info(
                f"Starting Feature Selection: {self.fs.name} for {self.n_features_to_select or 'all'} features on context {self.bac}"
            )
            # Feature selector's fit method should return a list of feature names
            # The ProteinDataset's _df already has the "Feature" column dropped if that was its role.
            # The target column is self.target.
            ranked_features = self.fs.fit(self._df.copy(), self.target)

            if self.n_features_to_select and self.n_features_to_select > 0:
                self._selected_features = ranked_features[: self.n_features_to_select]
            else:
                self._selected_features = ranked_features  # Use all ranked features if n_features_to_select is 0 or None

            self.logger.info(f"Selected features after FS: {self._selected_features}")
            if not self._selected_features:
                self.logger.warning("Feature selection resulted in no features.")
                return pd.DataFrame()
            return self._X[self._selected_features]

        self.logger.info(
            "No feature selector provided and no pre-selected features. Using all available features."
        )
        self._selected_features = self._X.columns.tolist()  # Update internal list
        return self._X  # Return all features

    def run(self) -> Dict[str, Any]:
        """
        Executes the protein data analysis run.

        Returns:
            Dict[str, Any]: A dictionary containing the run results and configuration.
        """
        result_data = self.run_config.copy()
        result_data.update(
            {
                "scores": [],
                "mean_cv_score": None,
                "model_summary": {},
                "best_params": {},
                "test_accuracy": None,
                "test_f1_score": None,
                "status": "Failed",  # Default status
                "error": None,
                "actual_n_features_used": 0,
                "final_selected_features_list": [],
            }
        )

        try:
            selected_X_df = self.selected_features_df
            target_y = self._y

            if selected_X_df.empty:
                self.logger.error(
                    "No features available for model training after selection."
                )
                result_data["error"] = "No features available after selection."
                result_data["status"] = "Failed"
                return result_data

            if target_y.empty:
                self.logger.error("Target variable is empty.")
                result_data["error"] = "Target variable is empty."
                result_data["status"] = "Failed"
                return result_data

            if selected_X_df.shape[0] != target_y.shape[0]:
                self.logger.error(
                    f"Shape mismatch between features ({selected_X_df.shape}) and target ({target_y.shape})."
                )
                result_data["error"] = "Shape mismatch between features and target."
                result_data["status"] = "Failed"
                return result_data

            result_data["actual_n_features_used"] = selected_X_df.shape[1]
            result_data["final_selected_features_list"] = selected_X_df.columns.tolist()
            self.run_config["selected_features"] = (
                selected_X_df.columns.tolist()
            )  # Update run_config with actual features used

            self.logger.info(
                f"Selected features DataFrame head: {selected_X_df.head()}"
            )

            # Save to csv for debugging
            # mt_with_target = pd.concat([selected_X_df, target_y.rename(self.target)], axis=1)
            # debug_filename = f"debug_protein_data_{self.bac}_{self.target.replace('/', '_')}_{self.model.__class__.__name__}.csv"
            # mt_with_target.to_csv(debug_filename, index=False)
            # self.logger.info(f"Saved debug data to {debug_filename}")

            self.logger.info(
                f"Fitting Model (type: {self.model.model_type}) with {selected_X_df.shape[1]} features."
            )

            scaler = StandardScaler()
            scaled_X_df = pd.DataFrame(
                scaler.fit_transform(selected_X_df),
                columns=selected_X_df.columns,
                index=selected_X_df.index,
            )
            self.logger.info("Data scaled.")

            # --- Train/Test split for test metrics ---
            # Ensure target_y has enough samples for stratification if it's multi-class
            stratify_option = (
                target_y
                if len(target_y.unique()) > 1
                and target_y.nunique() < len(target_y) // 2
                else None
            )
            if (
                stratify_option is not None and stratify_option.value_counts().min() < 2
            ):  # Check for scikit-learn min samples per class for stratify
                self.logger.warning(
                    f"Target stratification might not be possible due to small class sizes. Disabling stratification. Value counts: {target_y.value_counts().to_dict()}"
                )
                stratify_option = None

            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X_df,
                target_y,
                test_size=0.2,
                random_state=42,
                stratify=stratify_option,
            )
            self.logger.info(
                f"Data split into train/test. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
            )

            self.model.fit(X_train, y_train)
            self.logger.info("Model fitted on training data.")

            y_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            self.logger.info(
                f"Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}"
            )

            self.logger.info(f"Cross-validating Model (type: {self.model.model_type})")
            # Note: cross_validate in your model wrapper might handle scaling internally or expect scaled data.
            # Here, we pass the scaled_X_df.
            scores = self.model.cross_validate(scaled_X_df, target_y)
            self.logger.info(f"Cross-validation scores: {scores}")

            model_summary = self.model.get_summary()
            best_params = self.model.get_best_params()

            result_data["scores"] = (
                scores.tolist() if isinstance(scores, np.ndarray) else scores
            )
            result_data["mean_cv_score"] = (
                np.mean(scores)
                if scores is not None
                and (
                    isinstance(scores, list)
                    and len(scores) > 0
                    or isinstance(scores, np.ndarray)
                    and scores.size > 0
                )
                else None
            )
            result_data["model_summary"] = model_summary
            result_data["best_params"] = best_params
            result_data["test_accuracy"] = test_accuracy
            result_data["test_f1_score"] = test_f1
            result_data["status"] = "Success"
            self.logger.info(
                f"Run Success. Mean CV Score: {result_data['mean_cv_score']}"
            )

        except Exception as e:
            self.logger.error(
                f"Error during model run for config {self.run_config}: {e}",
                exc_info=True,
            )
            result_data["error"] = str(e)
            # Status remains 'Failed'

        return result_data
