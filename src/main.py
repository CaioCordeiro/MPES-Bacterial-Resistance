import argparse  # Make sure this import is at the top of the file
import os  # Import os if needed for path manipulation
import time  # Import time for timestamping plots if needed
from typing import Any, Dict, List
import matplotlib.pyplot as plt  # Make sure this is imported
import numpy as np  # Make sure this is imported

from data_providers.genetic_data.genetic_data import \
    GeneticDataset  # Make sure this is imported
from models.logistic_regression_model import \
    LogisticRegressionModel  # Make sure this is imported
from models.svm import SupportVectorMachineModel

from feature_selection.pearson_correlation import \
    PearsonCorrelationSelector  # Make sure this is imported
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from feature_selection.shap_feature_selector import ShapFeatureSelector
from feature_selection.rrelief import RReliefF
from pipeline.pipeline import Pipeline
from runs.run_results import \
    save_final_results  # Import if saving here instead of Pipeline
from utils.logging_config import (get_logger,  # Make sure this is imported
                                  setup_logging)


# --- Functions: check_and_prepare_data, prepare_all_datasets ---
# Ensure these functions are defined here or imported correctly
# Example placeholder definitions if they are missing:
def check_and_prepare_data(bac_name, max_sra_ids):
    logger = get_logger()  # Assuming get_logger is available
    logger.info(f"Checking/Preparing data for {bac_name} (max_ids={max_sra_ids})...")
    # Placeholder: Add actual data preparation logic using GeneticDataset
    try:
        dataset = GeneticDataset(bac_name=bac_name, max_sra_ids=max_sra_ids, root_dir='/mnt/d')
        # Trigger data loading/generation if not cached
        _ = dataset.treated_data
        logger.info(f"Data preparation successful for {bac_name}.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to prepare data for {bac_name}: {e}", exc_info=True)
        return None


def prepare_all_datasets(bacteria_list, max_sra_ids):
    logger = get_logger()  # Assuming get_logger is available
    prepared_datasets = {}
    start_time = time.time()
    for bac in bacteria_list:
        logger.info(f"Preparing dataset for {bac}...")
        dataset = check_and_prepare_data(bac, max_sra_ids)
        if dataset:
            prepared_datasets[bac] = dataset
            end_time = time.time()
            logger.info(
                f"Dataset for {bac} prepared successfully in {end_time - start_time:.2f} seconds"
            )
        else:
            logger.error(f"Failed to prepare dataset for {bac}.")
        start_time = time.time()  # Reset timer for next bacteria
    return prepared_datasets


def plot_metrics(results: List[Dict[str, Any]], bacteria: str, timestamp: str = None):
    """
    Plots metrics for each combination of model and feature selector for a given bacteria.
    Saves a separate plot for each (model, feature selector, bacteria) combination.
    """
    import os

    # Group results by (model, feature selector)
    grouped = {}
    for result in results:
        if result.get("bacteria") == bacteria:
            model_name = result.get("model_class", "UnknownModel")
            fs_name = result.get("feature_selector_class", "NoFS")
            key = (model_name, fs_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

    for (model_name, fs_name), group_results in grouped.items():
        metrics = {
            "f1_score": [],
            "accuracy": [],
            "n_features": [],
            "scores": [],
        }
        for result in group_results:
            metrics["n_features"].append(result.get("n_features_requested"))
            metrics["f1_score"].append(result.get("test_f1_score", 0))
            metrics["accuracy"].append(result.get("test_accuracy", 0))
            metrics["scores"].append(
                np.mean(result.get("scores", [])) if result.get("scores") else 0
            )

        feature_sizes = sorted(set(metrics["n_features"]))
        plt.figure(figsize=(10, 6))
        # Plot F1 Score
        plt.plot(
            feature_sizes,
            [metrics["f1_score"][metrics["n_features"].index(n)] for n in feature_sizes],
            marker='o',
            label='F1 Score'
        )
        # Plot Accuracy
        plt.plot(
            feature_sizes,
            [metrics["accuracy"][metrics["n_features"].index(n)] for n in feature_sizes],
            marker='o',
            label='Accuracy'
        )

        plt.title(f'{bacteria} - {model_name} - {fs_name}')
        plt.xlabel('Number of Features')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{bacteria}_{model_name}_{fs_name}_metrics.jpg"
        plt.savefig(filename)
        plt.close()
        print(f"Metrics plot saved for {bacteria}, {model_name}, {fs_name} as {filename}")

def main():
    # --- Argument Parser Setup --- <<< ADD THIS SECTION
    parser = argparse.ArgumentParser(description="Run Genetic Analysis Pipeline")
    parser.add_argument(
        "--max_sra_ids",
        type=int,
        default=None,
        help="Maximum number of SRA IDs to process per bacteria (default: None, process all available).",
    )
    parser.add_argument(
        "--bacteria",
        nargs="+",  # Allows multiple bacteria names
        default=["kleb"],  # Default bacteria if none provided
        help="List of bacteria names to process (e.g., kleb ech sa).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional file path to save logs (default: None, log to console).",
    )
    # --- End Argument Parser Setup ---

    args = parser.parse_args()  # This line should now work
    print(args)
    logger = setup_logging(args.log_level, f"{time.strftime('%Y%m%d-%H:%M:%S')}_{args.log_file}")
    logger.info(f"Starting pipeline with log level: {args.log_level}")

    bacteria_list = args.bacteria
    # Define models and feature selectors (ensure they are lists of classes)
    models_to_run = [
                    SupportVectorMachineModel, 
                     LogisticRegressionModel
                     ]  # Add others like SVC if needed
    feature_selectors_to_run = [
        PearsonCorrelationSelector,
        # BanzhafFeatureSelector,
        ShapFeatureSelector,
        RReliefF
    ]  # Add others like RReliefF if needed

    # Use a smaller range for faster testing initially
    # feature_range_to_run = range(5, 11, 5) # e.g., 5, 10
    feature_range_to_run = range(5, 1023, 10)  # Original range
    target_antibiotic = "ciprofloxacin"  # Or get from args
    # Create a more robust results filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"genetic_run_results_{'_'.join(bacteria_list)}_{timestamp}.json"

    prepared_datasets = prepare_all_datasets(bacteria_list, args.max_sra_ids)
    if not prepared_datasets:
        logger.error("No valid bacteria datasets available. Exiting.")
        return
    # Paralelize the pipeline run
    
    pipeline = Pipeline(
        prepared_datasets=prepared_datasets,
        bacteria_list=[
            bac for bac in bacteria_list if bac in prepared_datasets
        ],  # Only use prepared ones
        max_sra_ids=args.max_sra_ids,
        models=models_to_run,
        feature_selectors=feature_selectors_to_run,
        feature_range=feature_range_to_run,
        target_antibiotic=target_antibiotic,
        results_output_file=results_file,  # Pass the output filename
    )

    pipeline.run_all()  # This now collects results and saves them at the end

    # Results are already saved by pipeline.run_all()
    # If you need the results list here, you can access it:
    final_results = pipeline.all_results
    # print(final_results)
    # logger.info(f"Pipeline finished. Results saved to {results_file}")

    # Plot results for each bacteria
    for bac in bacteria_list:
        if bac in prepared_datasets:  # Only plot for bacteria that were processed
            bac_results = [r for r in final_results if r.get("bacteria") == bac]
            print(bac_results)
            if bac_results:
                # plot_results(
                #     bac_results,
                #     f"Results for {bac.upper()} - Target: {target_antibiotic}",
                # )
                plot_metrics(bac_results, bac)  # Call to plot metrics
            else:
                logger.warning(f"No results found to plot for bacteria: {bac}")


if __name__ == "__main__":
    main()
