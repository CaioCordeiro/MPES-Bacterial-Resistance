import argparse  # Make sure this import is at the top of the file
import os  # Import os if needed for path manipulation
import time  # Import time for timestamping plots if needed
from typing import Any, Dict, List
import matplotlib.pyplot as plt  # Make sure this is imported
import numpy as np  # Make sure this is imported

from data_providers.genetic_data.genetic_data import \
    GeneticDataset  # Make sure this is imported
from models.linear_regression_model import \
    LinearRegressionModel  # Make sure this is imported
from models.svm import SupportVectorMachineModel
from feature_selection.pearson_correlation import \
    PearsonCorrelationSelector  # Make sure this is imported
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from feature_selection.shap_feature_selector import ShapFeatureSelector
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


# --- End Placeholder Functions ---


# --- Modify plot_results if needed based on the new dictionary structure ---
def plot_results(results: List[Dict[str, Any]], title: str):
    # Filter out failed runs if necessary
    # Ensure 'scores' exist and are not empty lists
    successful_results = [
        r
        for r in results
        if r.get("status") == "Success" and r.get("scores") and len(r["scores"]) > 0
    ]
    if not successful_results:
        print(f"No successful results with valid scores to plot for {title}")
        return

    # Ensure 'n_features_requested' exists
    feature_ranges = sorted(
        list(
            set(
                r["n_features_requested"]
                for r in successful_results
                if "n_features_requested" in r
            )
        )
    )
    models = sorted(
        list(set(r["model_class"] for r in successful_results if "model_class" in r))
    )
    feature_selectors = sorted(
        list(
            set(
                r["feature_selector_class"]
                for r in successful_results
                if "feature_selector_class" in r
            )
        )
    )

    # Determine subplot layout
    n_rows = len(models)
    n_cols = len(feature_selectors)
    if n_rows == 0 or n_cols == 0:
        print("No models or feature selectors found in results.")
        return

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )  # Ensure axes is always 2D
    fig.suptitle(title, fontsize=16)

    for i, model in enumerate(models):
        for j, fs in enumerate(feature_selectors):
            ax = axes[i, j]
            # Collect scores for this specific model/fs combination across feature counts
            plot_data = {}
            for n_features in feature_ranges:
                # Get scores for this specific point (model, fs, n_features)
                # Ensure all keys exist before accessing
                scores_list = [
                    r["scores"]
                    for r in successful_results
                    if r.get("model_class") == model
                    and r.get("feature_selector_class") == fs
                    and r.get("n_features_requested") == n_features
                ]
                # scores_list will be a list of lists (one list per run configuration)
                # We need to flatten it or handle the boxplot input correctly
                # Boxplot expects a list of score lists, one for each box
                if scores_list:
                    # Assuming scores is a list of floats from cross-validation
                    # If multiple runs had the exact same config (unlikely here),
                    # scores_list might have multiple inner lists. Boxplot handles this.
                    # Check if the first list of scores is not empty before adding
                    if scores_list[0]:
                        plot_data[n_features] = scores_list[
                            0
                        ]  # Take the first list if only one run per config point

            if plot_data:
                # Prepare data for boxplot: list of score lists, positions
                bp_data = [plot_data[n] for n in feature_ranges if n in plot_data]
                bp_positions = [n for n in feature_ranges if n in plot_data]
                if bp_data:
                    # Handle case where there's only one box
                    box_widths = 1.0  # Default width for a single box
                    if len(bp_positions) > 1:
                        # Calculate widths based on differences, handle potential zero diff
                        diffs = np.diff(bp_positions)
                        if len(diffs) > 0 and np.all(diffs > 0):
                            box_widths = min(diffs) * 0.5
                        elif (
                            len(diffs) > 0
                        ):  # If diffs exist but some are zero or negative (unlikely for range)
                            box_widths = (
                                np.mean(diffs[diffs > 0]) * 0.5
                                if np.any(diffs > 0)
                                else 1.0
                            )
                        # else: keep default width

                    ax.boxplot(
                        bp_data,
                        positions=bp_positions,
                        widths=box_widths,
                        showfliers=False,
                    )  # Hide outliers for cleaner plot
                    ax.set_xticks(feature_ranges)  # Ensure all ticks are shown
                    ax.set_xticklabels(feature_ranges)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No Data",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            ax.set_title(f"{model} - {fs}", fontsize=10)
            ax.set_xlabel("Number of Features", fontsize=8)
            # Determine Y-axis label based on scoring (assuming neg_mean_squared_error)
            # You might need to adjust this if using different scoring
            y_label = "CV Score (neg_mean_squared_error)"
            ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.6)  # Add grid for readability

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    # Save the plot instead of showing directly if running non-interactively
    # Sanitize title for filename
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    plot_filename = f"results_plot_{safe_title}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)  # Close the figure to free memory
    # plt.show() # Comment out if running in a non-GUI environment


def plot_metrics(results: List[Dict[str, Any]], bacteria: str):
    metrics = {
        "f1_score": [],
        "accuracy": [],
        "n_features": []
    }

    for result in results:
        if result.get("bacteria") == bacteria:
            metrics["n_features"].append(result.get("n_features_requested"))
            metrics["f1_score"].append(result.get("model_summary", {}).get("f1_score", 0))
            metrics["accuracy"].append(result.get("model_summary", {}).get("accuracy", 0))

    feature_sizes = sorted(set(metrics["n_features"]))
    
    plt.figure(figsize=(10, 6))
    
    # Plot F1 Score
    plt.plot(feature_sizes, [metrics["f1_score"][metrics["n_features"].index(n)] for n in feature_sizes], marker='o', label='F1 Score')
    
    # Plot Accuracy
    plt.plot(feature_sizes, [metrics["accuracy"][metrics["n_features"].index(n)] for n in feature_sizes], marker='o', label='Accuracy')

    plt.title(f'Metrics for {bacteria}')
    plt.xlabel('Number of Features')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid()
    
    plt.savefig(f'{bacteria}_metrics.jpg')
    plt.close()

    print(f"Metrics plot saved for {bacteria} as {bacteria}_metrics.jpg")


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
    models_to_run = [SupportVectorMachineModel]  # Add others like SVC if needed
    feature_selectors_to_run = [
        # PearsonCorrelationSelector,
        # BanzhafFeatureSelector,
        ShapFeatureSelector
    ]  # Add others like RReliefF if needed

    # Use a smaller range for faster testing initially
    # feature_range_to_run = range(5, 11, 5) # e.g., 5, 10
    feature_range_to_run = range(5, 200, 10)  # Original range
    target_antibiotic = "ciprofloxacin"  # Or get from args
    # Create a more robust results filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"genetic_run_results_{'_'.join(bacteria_list)}_{timestamp}.json"

    prepared_datasets = prepare_all_datasets(bacteria_list, args.max_sra_ids)
    if not prepared_datasets:
        logger.error("No valid bacteria datasets available. Exiting.")
        return

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
    logger.info(f"Pipeline finished. Results saved to {results_file}")

    # Plot results for each bacteria
    for bac in bacteria_list:
        if bac in prepared_datasets:  # Only plot for bacteria that were processed
            bac_results = [r for r in final_results if r.get("bacteria") == bac]
            if bac_results:
                plot_results(
                    bac_results,
                    f"Results for {bac.upper()} - Target: {target_antibiotic}",
                )
                plot_metrics(bac_results, bac)  # Call to plot metrics
            else:
                logger.warning(f"No results found to plot for bacteria: {bac}")


if __name__ == "__main__":
    main()
