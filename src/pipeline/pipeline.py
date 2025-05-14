import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Type, Union

import psutil
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.svm import SVC

from data_providers.genetic_data.genetic_data import GeneticDataset
from models.logistic_regression_model import LogisticRegressionModel
from models.svm import SupportVectorMachineModel
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from feature_selection.shap_feature_selector import ShapFeatureSelector
from feature_selection.pearson_correlation import PearsonCorrelationSelector
from feature_selection.rrelief import RReliefF
from runs.genetic_data_run import GeneticDataRun
from runs.run_results import save_final_results
from utils.logging_config import get_logger
import constants.constants as const


class Pipeline:
    """
    Pipeline to run experiments...
    Collects results centrally and saves at the end.
    """

    def __init__(
        self,
        prepared_datasets: Dict[str, GeneticDataset] = None,
        bacteria_list: List[str] = None,
        models: List[
            Union[
                Type[SVC], Type[SklearnLinearRegression], Type[LogisticRegressionModel]
            ]
        ] = None,
        feature_selectors: List[Type] = None,
        feature_range: range = range(5, 36, 5),
        target_antibiotic: str = "ciprofloxacin",
        max_sra_ids: Optional[int] = None,
        results_output_file: str = "genetic_run_results.json",
    ):
        self.prepared_datasets = prepared_datasets or {}
        self.bacteria_list = bacteria_list or ["sa", "ech", "kleb"]
        self.models = models if models is not None else [SVC, LogisticRegressionModel]
        self.feature_selectors = (
            feature_selectors
            if feature_selectors is not None
            else [PearsonCorrelationSelector, RReliefF]
        )
        self.feature_range = feature_range
        self.target_antibiotic = target_antibiotic
        self.max_sra_ids = max_sra_ids
        self.logger = get_logger()
        self.all_results: List[Dict[str, Any]] = []
        self.results_output_file = results_output_file

        available_cores = max(1, int(cpu_count() * 0.7))
        self.n_jobs = min(available_cores, 4)

    def run_all(self):
        """
        Runs all combinations... collecting results centrally.
        Saves all results to a file at the very end.
        """
        for bac in self.bacteria_list:
            if bac not in self.prepared_datasets:
                self.logger.warning(f"No prepared dataset for {bac}, skipping.")
                continue

            self._run_for_bacteria(bac, self.prepared_datasets[bac])

        self.logger.info(
            f"Finished all runs. Saving {len(self.all_results)} results..."
        )
        save_final_results(self.all_results, self.results_output_file)

    def _run_for_bacteria(self, bac, data):
        """
        Run pipeline for a specific bacteria, collecting results.
        Feature selection is performed once per model and feature selector combination.
        """
        self.logger.info(f"Running pipeline for {bac}...")
        df = data.treated_data.copy()
        anti_list_without_target = const.ANTIBIOTIC_LIST.copy()
        if self.target_antibiotic in anti_list_without_target:
            anti_list_without_target.remove(self.target_antibiotic)
        df = df.drop(anti_list_without_target, axis=1, errors="ignore")

        for ModelClass in self.models:
            # Train the model once if SHAP is used
            trained_model = None

            for FSClass in self.feature_selectors:
                # Perform feature selection
                try:
                    if FSClass == ShapFeatureSelector:
                        # Train the model if not already trained
                        fs_instance = FSClass(model=ModelClass())

                        ranked_features = fs_instance.fit(df, self.target_antibiotic)
                    else:
                        # Run other feature selection algorithms
                        fs_instance = FSClass()
                        ranked_features = fs_instance.fit(df, self.target_antibiotic)

                    self.logger.info(
                        f"Feature ranking completed for {bac} using {FSClass.__name__}. "
                        f"Ranked {len(ranked_features)} features."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Feature ranking failed for {bac} using {FSClass.__name__}: {e}",
                        exc_info=True,
                    )
                    continue

                # Train models with different numbers of features
                for n_features in self.feature_range:
                    # Get top n features
                    selected_features = ranked_features[:n_features]
                    self.logger.info(
                        f"Running {ModelClass.__name__} with {FSClass.__name__} "
                        f"for {bac} using {n_features} features."
                    )
                    # reduced_data = data.treated_data[
                    #     selected_features + [self.target_antibiotic]
                    # ]
                    self._run_single_configuration(
                        ModelClass, FSClass, n_features, bac, data, selected_features
                    )

    def _run_single_configuration(
        self, ModelClass, FSClass, n_features, bac, data, selected_features
    ):
        """
        Train a model using the selected features and collect results.
        """
        logger = get_logger()
        try:
            # Instantiate the model
            model = ModelClass()
            # Train the model
            run = GeneticDataRun(
                data=data,
                target=self.target_antibiotic,
                bac=bac,
                model=model,
                fs=None,  # FS is already applied
                selected_features=selected_features,
            )
            result_dict = run.run()

            # Add metadata to the result
            result_dict.update(
                {
                    "bacteria": bac,
                    "model_class": ModelClass.__name__,
                    "feature_selector_class": FSClass.__name__,
                    "n_features_requested": n_features,
                }
            )
            self.all_results.append(result_dict)

        except Exception as e:
            logger.error(
                f"Error in configuration (Bac={bac}, Model={ModelClass.__name__}, "
                f"FS={FSClass.__name__}, Feats={n_features}): {e}",
                exc_info=True,
            )
