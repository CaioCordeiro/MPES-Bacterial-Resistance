from typing import Any, Dict, List, Optional

import numpy as np  # Import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import constants.constants as const
from data_providers.genetic_data.genetic_data import GeneticDataset
from entities.interfaces.feature_selection import FeatureSelectionInterface
# from .run_results import RunResults # No longer needed here
from utils.logging_config import get_logger


class GeneticDataRun:
    """
    Encapsulates a single run of a genetic data analysis pipeline.
    Returns results as a dictionary.
    """

    def __init__(
        self,
        data: GeneticDataset,
        target: str,
        bac: str,
        model: BaseEstimator,
        fs: Optional[FeatureSelectionInterface] = None,
        run_config: Optional[Dict[str, Any]] = None,
        selected_features: Optional[List[str]] = None,
        n_features_to_select: Optional[int] = None,
    ):

        self.logger = get_logger()
        self.data: GeneticDataset = data
        self.target: str = target
        self.bac: str = bac
        self.model: BaseEstimator = model
        self.fs: FeatureSelectionInterface = fs
        self._selected_features: List[str] = selected_features if selected_features else []
        # self.results_store: RunResults = RunResults() # Removed
        self.n_features_to_select: int = n_features_to_select if n_features_to_select else 0
        self.run_config: Dict[str, Any] = run_config if run_config is not None else {}
        # Base config - more details added in run()
        self.run_config.update(
            {
                "database": data.name,
                "target": target,
                "bacteria": bac,
                "model_class": model.__class__.__name__,  # Store class name
                "feature_selection": fs.name if fs else None,
                "n_features_requested": getattr(
                    fs, "n_features_to_select", n_features_to_select
                ),  # Store requested features
            }
        )
        # from models.svm import SupportVectorMachineModel
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2, random_state=42)
        # self.model.fit(X_train, y_train)
        # print(model.get_summary())
        # # Accuracy on the test set
        # y_pred = model.predict(X_test)
        # accuracy = model.model.score(X_test, y_test)
        # print(f"Test set accuracy: {accuracy}")
        # # F1 score on the test set
        # from sklearn.metrics import f1_score
        # f1 = f1_score(y_test, y_pred, average='weighted')
        # print(f"F1 score on the test set: {f1}")
        # scores = model.cross_validate(self._X, self._y, cv=10)
        # print(f"Cross-validation scores: {scores}")
        # print(f"Mean cross-validation score: {scores.mean()}")
        # self.results_store.set_run_config(self.run_config) # Removed

    @property
    def _df(self):
        """
        Returns the DataFrame with the target variable and features.
        Excludes other antibiotic resistance columns.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        anti_list_without_target = const.ANTIBIOTIC_LIST.copy()
        # self.logger.debug(f"Antibiotic list: {const.ANTIBIOTIC_LIST}")
        if self.target in anti_list_without_target:
            anti_list_without_target.remove(self.target)
        return self.data.treated_data.drop(
            anti_list_without_target, axis=1, errors="ignore"
        ).dropna()

    @property
    def _X(self):
        """
        Returns the feature matrix (independent variables).

        Returns:
            pd.DataFrame: The feature matrix.
        """
        return self._df.drop(self.target, axis=1, errors="ignore")

    @property
    def _y(self):
        """
        Returns the target vector (dependent variable).

        Returns:
            pd.Series: The target vector.
        """
        return self._df[self.target]

    @property
    def selected_features(self):
        """
        Performs feature selection and returns the DataFrame with selected features.

        Returns:
            pd.DataFrame: The DataFrame containing only the selected features.
        """
        if self._selected_features is None or len(self._selected_features) == 0:
            self.logger.info(
                f"Starting Feature Selection: {self.fs.name} for {self.n_features_to_select} features on bacteria {self.bac}"
            )
            # Ensure fit returns the DataFrame with selected features
            self._selected_features = self.fs.fit(
                self._df.copy(), self.target
            )[:self.n_features_to_select]  # Pass a copy to avoid modifying original _df if fs modifies inplace
        return self._X[self._selected_features]

    def run(self) -> Dict[str, Any]:  # Changed return type
        """
        Executes the genetic data analysis run.

        Returns:
            Dict[str, Any]: A dictionary containing the run results and configuration.
        """
        selected_X_df = (
            self.selected_features
        )  # Get the DataFrame with selected features
        # show the selected df
        self.logger.info(
            f"Selected features DataFrame: {selected_X_df.head()}"
        )
        target_y = self._y  # Get the target series
        # Save to csv for debugging the full master table, with target (sekected_X_df  + target_y)
        mt_with_target = pd.concat(
            [selected_X_df, target_y], axis=1
        )
        mt_with_target.to_csv(
            f"mt_with_target_{self.bac}_{self.target}.csv",
            index=False,
        )

        # Prepare results dictionary
        result_data = self.run_config.copy()  # Start with base config
        result_data["scores"] = []
        result_data["model_summary"] = {}
        result_data["best_params"] = {}
        result_data["status"] = "Failed"  # Default status

        try:
            self.logger.info(f"Fitting Model (type: {self.model.model_type})")
           # --- Cross-validation ---
            scaler = StandardScaler()
            selected_X_df = pd.DataFrame(
                scaler.fit_transform(selected_X_df),
                columns=selected_X_df.columns,
                index=selected_X_df.index,
            )
            # self.logger.info(f"Scaled DataFrame: {selected_X_df.head()}")
            self.logger.info("Done cross validating model")
            # Scale df
            # --- Train/Test split for test metrics ---
            X_train, X_test, y_train, y_test = train_test_split(
                selected_X_df, target_y, test_size=0.2, random_state=42, stratify=target_y if len(set(target_y)) > 1 else None
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.logger.info(f"Cross-validating Model (type: {self.model.model_type})")
            scores = self.model.cross_validate(selected_X_df, target_y)
            self.logger.info(
                f"Cross-validation scores: {scores}"
            )  # Log the cross-validation scores

            model_summary = self.model.get_summary()
            best_params = self.model.get_best_params()

            result_data["scores"] = (
                scores.tolist() if isinstance(scores, np.ndarray) else scores
            )
            result_data["mean_cv_score"] = (
                np.mean(scores) if scores is not None and len(scores) > 0 else None
            )
            result_data["model_summary"] = model_summary
            result_data["best_params"] = best_params
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
            # Keep status as 'Failed'
        # print(
        #     f"Run results: {result_data}"
        # )
        return result_data  # Return the results dictionary
