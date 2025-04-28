from typing import Any, Dict, List, Optional

import numpy as np  # Import numpy
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score

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
        fs: FeatureSelectionInterface,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        if not hasattr(data, "treated_data") or data.treated_data is None:
            raise ValueError(
                "Dataset must be fully prepared before passing to GeneticDataRun"
            )

        self.logger = get_logger()
        self.data: GeneticDataset = data
        self.target: str = target
        self.bac: str = bac
        self.model: BaseEstimator = model
        self.fs: FeatureSelectionInterface = fs
        # self.results_store: RunResults = RunResults() # Removed
        self.run_config: Dict[str, Any] = run_config if run_config is not None else {}
        # Base config - more details added in run()
        self.run_config.update(
            {
                "database": data.name,
                "target": target,
                "bacteria": bac,
                "model_class": model.__class__.__name__,  # Store class name
                "feature_selection": fs.name,
                "n_features_requested": getattr(
                    fs, "n_features_to_select", None
                ),  # Store requested features
            }
        )
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
        self.logger.debug(f"Antibiotic list: {const.ANTIBIOTIC_LIST}")
        if self.target in anti_list_without_target:
            anti_list_without_target.remove(self.target)
        return self.data.treated_data.drop(
            anti_list_without_target, axis=1, errors="ignore"
        )

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
    def _selected_features(self):
        """
        Performs feature selection and returns the DataFrame with selected features.

        Returns:
            pd.DataFrame: The DataFrame containing only the selected features.
        """
        self.logger.info(
            f"Starting Feature Selection: {self.fs.name} for {self.fs.n_features_to_select} features on bacteria {self.bac}"
        )
        # Ensure fit returns the DataFrame with selected features
        selected_data = self.fs.fit(
            self._df.copy(), self.target
        )  # Pass a copy to avoid modifying original _df if fs modifies inplace
        # Extract only the feature columns (X part) from the result of fit
        selected_X = selected_data.drop(columns=[self.target], errors="ignore")
        self.logger.info(
            f"Finished Feature Selection: {self.fs.name} on bacteria {self.bac}. Selected {selected_X.shape[1]} features."
        )
        return selected_X  # Return only the features DataFrame

    def run(self) -> Dict[str, Any]:  # Changed return type
        """
        Executes the genetic data analysis run.

        Returns:
            Dict[str, Any]: A dictionary containing the run results and configuration.
        """
        selected_X_df = (
            self._selected_features
        )  # Get the DataFrame with selected features
        target_y = self._y  # Get the target series

        # Prepare results dictionary
        result_data = self.run_config.copy()  # Start with base config
        result_data["scores"] = []
        result_data["model_summary"] = {}
        result_data["best_params"] = {}
        result_data["status"] = "Failed"  # Default status

        try:
            self.logger.info(f"Fitting Model (type: {self.model.model_type})")
            self.model.fit(selected_X_df, target_y)
            self.logger.info("Done Fitting model")
            # Use numpy mean for cross-validation scores
            scores = self.model.cross_validate(
                selected_X_df, target_y, cv=5
            )  # Reduced CV folds for speed
            self.logger.info("Done cross validating model")

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

        return result_data  # Return the results dictionary
