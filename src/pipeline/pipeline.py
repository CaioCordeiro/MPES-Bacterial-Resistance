# /home/ccmr/workspace/MPES-Bacterial-Resistance/src/pipeline/pipeline.py

import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Type, Union  # Added Any

import psutil
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.svm import SVC

import constants.constants as const
from data_providers.genetic_data.genetic_data import GeneticDataset
from entities.models.linear_regression_model import LinearRegressionModel
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from feature_selection.pearson_correlation import PearsonCorrelationSelector
from feature_selection.rrelief import RReliefF
from runs.genetic_data_run import GeneticDataRun
from runs.run_results import save_final_results  # Import the save function
from utils.logging_config import get_logger


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
            Union[Type[SVC], Type[SklearnLinearRegression], Type[LinearRegressionModel]]
        ] = None,
        feature_selectors: List[Type] = None,
        feature_range: range = range(5, 36, 5),  # Adjusted step back to 5
        target_antibiotic: str = "Ampicillin",
        max_sra_ids: Optional[int] = None,
        results_output_file: str = "genetic_run_results.json",  # Added output file path
    ):
        self.prepared_datasets = prepared_datasets or {}
        self.bacteria_list = bacteria_list or ["sa", "ech", "kleb"]
        # Ensure LinearRegressionModel is included if specified or default
        default_models = [SVC, LinearRegressionModel]
        self.models = models if models is not None else default_models
        # Ensure PearsonCorrelationSelector is included if specified or default
        default_fs = [
            # BanzhafFeatureSelector, # Uncomment if needed
            # RReliefF,             # Uncomment if needed
            PearsonCorrelationSelector,
        ]
        self.feature_selectors = (
            feature_selectors if feature_selectors is not None else default_fs
        )

        self.feature_range = feature_range
        self.target_antibiotic = target_antibiotic
        self.max_sra_ids = max_sra_ids
        self.logger = get_logger()
        self.all_results: List[Dict[str, Any]] = []  # Store all results here
        self.results_output_file = results_output_file  # Store output file path

        available_cores = max(1, int(cpu_count() * 0.7))
        self.n_jobs = min(available_cores, 4)  # Keep core limit

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

        # Save all collected results after all bacteria are processed
        self.logger.info(
            f"Finished all runs. Saving {len(self.all_results)} results..."
        )
        save_final_results(self.all_results, self.results_output_file)

    def _run_for_bacteria(self, bac, data):
        """Run pipeline for a specific bacteria, collecting results."""
        self.logger.info(f"Running pipeline for {bac}...")
        run_configs_for_pool = []

        # --- REMOVED: Pearson Pre-calculation Block ---
        # pearson_selector_instance = None
        # if PearsonCorrelationSelector in self.feature_selectors:
        #    ... (code to pre-calculate scores was here) ...
        # ----------------------------------------------

        for ModelClass in self.models:
            for FSClass in self.feature_selectors:
                # --- Loop through feature range ---
                for n_features in self.feature_range:
                    # Pass all necessary arguments for _run_single_configuration_wrapper
                    # We now pass the FS *Class* and n_features separately
                    run_configs_for_pool.append(
                        (
                            ModelClass,
                            FSClass,
                            n_features,
                            bac,
                            data,
                            self.target_antibiotic,
                        )
                    )

        if not run_configs_for_pool:
            self.logger.warning(f"No valid configurations to run for {bac}.")
            return

        optimal_processes = self._determine_optimal_processes()
        self.logger.info(
            f"Running {len(run_configs_for_pool)} configurations for {bac} with {optimal_processes} parallel processes."
        )

        results_for_bac = []
        try:
            with Pool(processes=optimal_processes) as pool:
                # Use starmap to pass multiple arguments
                results_for_bac = pool.starmap(
                    self._run_single_configuration_wrapper, run_configs_for_pool
                )
        except Exception as e:
            self.logger.error(
                f"Error during multiprocessing pool execution for {bac}: {e}",
                exc_info=True,
            )

        # Extend the main results list (filter out None results if errors occurred)
        self.all_results.extend([res for res in results_for_bac if res is not None])
        self.logger.info(
            f"Finished runs for {bac}. Collected {len(results_for_bac)} results."
        )

    def _determine_optimal_processes(self):
        """Calculates optimal number of processes based on CPU and memory"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # Estimate memory needed per process (adjust based on observation)
        # Start with a conservative estimate, e.g., 1.5-2GB
        memory_per_process_gb = 2.0
        memory_based_processes = max(
            1, int(available_memory_gb / memory_per_process_gb)
        )
        optimal_processes = max(
            1, min(self.n_jobs, memory_based_processes)
        )  # Ensure at least 1 process
        self.logger.debug(
            f"Available Mem: {available_memory_gb:.2f} GB, Mem/Process: {memory_per_process_gb} GB -> Mem-based processes: {memory_based_processes}"
        )
        self.logger.debug(
            f"CPU-based processes: {self.n_jobs}, Final Optimal Processes: {optimal_processes}"
        )
        return optimal_processes

    @staticmethod  # Make this static as it doesn't rely on Pipeline instance state
    def _run_single_configuration_wrapper(
        ModelClass, FSClass, n_features, bac, data, target_antibiotic
    ):
        """
        Wrapper function to instantiate model/fs correctly and call GeneticDataRun.run.
        Handles exceptions within the worker process.
        Instantiates FSClass freshly for each run.
        """
        logger = get_logger()  # Get logger within the worker
        try:
            # --- Instantiate Model ---
            if ModelClass == LinearRegressionModel:
                # Example: Defaulting to 'ridge'. Modify if other types needed per config.
                model = ModelClass(model_type="ridge")
            elif issubclass(ModelClass, (SVC, SklearnLinearRegression)):
                model = ModelClass()  # Default instantiation for sklearn models
            else:
                logger.error(f"Unsupported model class: {ModelClass.__name__}")
                return None  # Cannot proceed

            # --- Instantiate FS instance with n_features ---
            # Always instantiate fresh for each configuration run
            try:
                # Try constructor with n_features_to_select parameter
                fs_run_instance = FSClass(n_features_to_select=n_features)
            except TypeError:
                # If constructor doesn't take n_features, instantiate without
                # and set attribute if possible
                fs_run_instance = FSClass()
                if hasattr(fs_run_instance, "n_features_to_select"):
                    fs_run_instance.n_features_to_select = n_features
                else:
                    logger.warning(
                        f"{fs_run_instance.__class__.__name__} might not support n_features_to_select."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to instantiate {FSClass.__name__} with n_features={n_features}: {e}"
                )
                return None  # Cannot proceed if FS instantiation fails

            logger.info(
                f"Worker executing: Bac={bac}, Model={ModelClass.__name__}, "
                f"FS={fs_run_instance.__class__.__name__}, Feats={n_features}"
            )

            # --- Execute Run ---
            run = GeneticDataRun(
                data=data,
                target=target_antibiotic,
                bac=bac,
                model=model,
                fs=fs_run_instance,  # Use the freshly instantiated instance
            )
            result_dict = run.run()  # run() now returns a dictionary

            # Add top-level metadata for clarity in final results file
            result_dict["bacteria"] = bac
            result_dict["model_class"] = ModelClass.__name__
            result_dict["feature_selector_class"] = fs_run_instance.__class__.__name__
            result_dict["n_features_requested"] = n_features

            return result_dict

        except Exception as e:
            logger.error(
                f"Error in worker for config (Bac={bac}, Model={ModelClass.__name__}, FS={FSClass.__name__}, Feats={n_features}): {e}",
                exc_info=True,
            )
            # Return a dictionary indicating failure for this specific run
            return {
                "bacteria": bac,
                "model_class": ModelClass.__name__,
                "feature_selector_class": FSClass.__name__,
                "n_features_requested": n_features,
                "status": "Failed in Worker",
                "error": str(e),
                "scores": [],
                "model_summary": {},
                "best_params": {},
            }

    # Remove _instantiate_fs as logic is now fully in the wrapper
    # Remove get_results as results are saved directly in run_all
