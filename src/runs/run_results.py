import json
import os
import threading

from utils.logging_config import get_logger


def save_final_results(results_list: list, output_file="genetic_run_results.json"):
    """Saves the final list of all run results to a JSON file."""
    logger = get_logger()
    try:
        # Ensure directory exists if output_file includes a path
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")

        with open(output_file, "w") as f:
            json.dump(results_list, f, indent=4)
        logger.info(f"Final results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final results to {output_file}: {e}")


class RunResults:
    """
    A class to store and manage the results of data runs.
    Provides thread-safe storage and file I/O capabilities.
    """

    def __init__(self, output_file="genetic_run_results.json"):
        self._results = []
        self._lock = threading.Lock()
        self._output_file = output_file
        self.logger = get_logger()
        self._current_run_config = {}  # Instance-level config for the current run

    def store_result(self, scores: list, run_config: dict):
        """Stores the scores and configuration of a single run."""
        result = {"scores": list(scores)}
        result.update(run_config)
        with self._lock:
            self._results.append(result)
            self._save_results_to_file()
            self.logger.info(f"Saved file")

    def set_run_config(self, config: dict):
        """Sets the configuration for the current run."""
        self._current_run_config = config

    def get_current_run_config(self) -> dict:
        """Returns the configuration for the current run."""
        return self._current_run_config

    def get_all_results(self) -> list:
        """Returns a list of all stored run results."""
        with self._lock:
            return list(self._results)

    def save_results(self, filename: str = None):
        """Saves all stored run results to a JSON file."""
        filepath = filename if filename else self._output_file
        with self._lock:
            with open(filepath, "w") as f:
                json.dump(self._results, f, indent=4)
        self.logger.info(f"Results saved to: {filepath}")

    def load_results(self, filename: str = None):
        """Loads run results from a JSON file."""
        filepath = filename if filename else self._output_file
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self._results = json.load(f)
            self.logger.info(f"Results loaded from: {filepath}")
        else:
            self.logger.warning(f"File not found at {filepath}. No results loaded.")
            self._results = []

    def clear_results(self):
        """Clears all stored run results."""
        with self._lock:
            self._results = []
        self.logger.info("All stored results cleared.")

    def _save_results_to_file(self):
        """Internal method to save results to the default output file."""
        self.save_results(self._output_file)
