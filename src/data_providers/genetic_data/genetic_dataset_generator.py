import csv
import fcntl
import os
import os.path
import threading
import time
from datetime import datetime
from itertools import product
from typing import IO, Dict, Generator, List, Optional

import pandas as pd

from constants.constants import (ANTIBIOTIC_FILE, ANTIBIOTIC_LIST, ROOT_DIR,
                                 DATASET_OUTPUT_DIR, FEATURE_DIR, K_SIZE,
                                 MAX_MEMORY_PERCENT)
from utils.logging_config import get_logger

from .utils import (check_dataset_file,  # Assuming these are in 'utils.py'
                    create_folder, normalize_mic, write_csv_file)


class DatasetGenerator:
    """
    A class to generate a dataset by combining k-mer counts from feature files
    with antibiotic Minimum Inhibitory Concentration (MIC) data.
    """

    def __init__(self, root_dir: str = ROOT_DIR,
        feature_dir: str = FEATURE_DIR,
        output_dir: str = DATASET_OUTPUT_DIR,
        antibiotic_file: str = ANTIBIOTIC_FILE,
        antibiotic_list: List[str] = ANTIBIOTIC_LIST,
        k_size: int = K_SIZE,
        cache_dir: str = "data/cache",
        memory_limit_percent: float = MAX_MEMORY_PERCENT,
        bac_name: int = "kleb",
    ):
        """
        Initializes the DatasetGenerator.

        Args:
            root_dir: The root directory for all other files.
            feature_dir: The directory containing the k-mer count CSV files.
            output_dir: The directory where the generated dataset CSV will be saved.
            antibiotic_file: The path to the CSV file containing antibiotic MIC data.
            antibiotic_list: A list of antibiotics to include in the dataset.
                             Defaults to the list defined in constants.py.
            k_size: The size of the k-mers used to generate the feature files.
            memory_limit_percent: Maximum percentage of system memory to use
        """
        self.root_dir = root_dir
        self.bac_name = bac_name
        self.feature_dir = os.path.join(self.root_dir, feature_dir)
        self.output_dir = os.path.join(self.root_dir, output_dir)
        self.cache_dir = os.path.join(self.root_dir, cache_dir)
        self.lock_dir = os.path.join(self.cache_dir, "locks")
        self.antibiotic_file = os.path.join(self.root_dir, f"{antibiotic_file}/{self.bac_name}/antibotic_relation.csv")
        self.dataset_lock_file = os.path.join(
            self.lock_dir, f"{self.bac_name}_dataset.lock"
        )
        self.antibiotic_list = antibiotic_list
        self.k_size = k_size
        self.memory_limit_percent = memory_limit_percent
        create_folder(os.path.join(self.root_dir, self.output_dir))
        create_folder(os.path.join(self.root_dir, self.cache_dir))
        create_folder(os.path.join(self.root_dir, self.lock_dir))
        self.logger = get_logger()

    def _acquire_lock(self) -> Optional[IO]:
        """
        Acquires a lock for dataset generation to prevent parallel processing.

        Returns:
            Optional file descriptor if lock was acquired, None otherwise.
        """
        try:
            fd = open(self.dataset_lock_file, "w")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (IOError, BlockingIOError):
            # Another process is already generating the dataset
            return None

    def _release_lock(self, fd: Optional[IO]) -> None:
        """Releases a previously acquired lock."""
        if fd:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def _generate_kmer_combinations(
        self, alphabet: str = "ACGT"
    ) -> Generator[str, None, None]:
        """
        Generates all possible k-mer combinations of length k_size in lexicographical order.

        Args:
            alphabet: The set of characters to use in the k-mers (string).
                      Defaults to "ACGT".

        Yields:
            str: The next k-mer combination.
        """
        for combination_tuple in product(alphabet, repeat=self.k_size):
            yield "".join(combination_tuple)

    def _get_mic_data(
        self, seq_name: str, mic_columns: List[str] = None
    ) -> List[float]:
        """
        Retrieves and normalizes MIC data for a given sequence from the antibiotic file.

        Args:
            seq_name: The SRA identifier of the sequence.
            mic_columns: The column names in the antibiotic file.
                         Defaults to ["genome", "sra_id", "patric_id", "antibiotic", "mic_actual", "mic_predicted"].

        Returns:
            A list of normalized MIC values for the predefined antibiotic list.
        """
        if mic_columns is None:
            mic_columns = [
                "sra_id",
                "antibiotic",
                "mic_actual",
            ]
        try:
            csv_data = pd.read_csv(self.antibiotic_file, header=0, names=mic_columns)
            sra_relation = csv_data.loc[csv_data["sra_id"] == seq_name][
                ["antibiotic", "mic_actual"]
            ]
            sra_relation = sra_relation.set_index("antibiotic")
            actual_mic = sra_relation.to_dict()["mic_actual"]
            mic_list = []
            for antibiotic in self.antibiotic_list:
                mic_list.append(actual_mic.get(antibiotic.lower(), 0))
            self.logger.debug(f"MIC data for {seq_name}: {mic_list}")
            return mic_list
        except FileNotFoundError:
            self.logger.error(
                f"Error: Antibiotic relation file not found at {self.antibiotic_file}"
            )
            return [0.0] * len(self.antibiotic_list)
        except KeyError as e:
            self.logger.error(
                f"Error: Column '{e}' not found in {self.antibiotic_file}"
            )
            return [0.0] * len(self.antibiotic_list)

    def _get_phenotype_data(
        self, seq_name: str, phenotype_columns: List[str] = None
    ) -> List[float]:
        """
        Retrieves and normalizes Resistant Phenotype data for a given sequence from the antibiotic file.

        Args:
            seq_name: The SRA identifier of the sequence.
            phenotype_columns: The column names in the antibiotic file.
                         Defaults to ["genome", "sra_id", "patric_id", "antibiotic", "Resistant Phenotype", "mic_predicted"].

        Returns:
            A list of normalized MIC values for the predefined antibiotic list.
        """
        if phenotype_columns is None:
            phenotype_columns = [
                "sra_id",
                "antibiotic",
                "Resistant Phenotype",
            ]
        try:
            csv_data = pd.read_csv(
                self.antibiotic_file, header=0, names=phenotype_columns
            )
            sra_relation = csv_data.loc[csv_data["sra_id"] == seq_name][
                ["antibiotic", "Resistant Phenotype"]
            ]
            sra_relation = sra_relation.set_index("antibiotic")
            actual_phenotype = sra_relation.to_dict()["Resistant Phenotype"]
            phenotype_list = []
            for antibiotic in self.antibiotic_list:
                phenotype_list.append(actual_phenotype.get(antibiotic.lower(), 0))
            # phenotype_list = [0 if p == "Susceptible" else 1 for p in phenotype_list]
            self.logger.debug(f"Phenotype data for {seq_name}: {phenotype_list}")
            return phenotype_list
        except FileNotFoundError:
            self.logger.error(
                f"Error: Antibiotic relation file not found at {self.antibiotic_file}"
            )
            return [0.0] * len(self.antibiotic_list)
        except KeyError as e:
            self.logger.error(
                f"Error: Column '{e}' not found in {self.antibiotic_file}"
            )
            return [0.0] * len(self.antibiotic_list)

    def _get_kmer_counts(self, file_name: str) -> List[int]:
        """
        Reads k-mer counts from a CSV file.

        Args:
            file_name: The name of the CSV file containing k-mer counts.

        Returns:
            A list of k-mer counts.
        """
        filepath = os.path.join(self.root_dir, self.feature_dir, file_name)
        try:
            with open(filepath, "r") as csvfile:
                csvreader = csv.reader(csvfile)
                row = [int(item) for item in next(csvreader)]
                return row
        except FileNotFoundError:
            self.logger.warning(f"Feature file not found: {filepath}")
            return []
        except StopIteration:
            self.logger.warning(f"Feature file is empty: {filepath}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading feature file {filepath}: {e}")
            return []

    def generate_dataset(self) -> None:
        """
        Generates the combined dataset by iterating through feature files,
        retrieving k-mer counts and phenotype data, and saving the result to a CSV file.
        """
        file_list = os.listdir(self.feature_dir)
        seq_data: List[List] = []

        # Output file path
        output_filepath = os.path.join(self.root_dir, self.output_dir, f"{self.bac_name}_dataset.csv")

        # Try to acquire lock for dataset generation
        lock_fd = self._acquire_lock()
        if not lock_fd:
            # Another process is already generating the dataset
            self.logger.info(
                f"Another process is already generating the dataset for {self.bac_name}. Waiting..."
            )
            # Wait for the dataset file to appear
            max_wait_time = 600  # 10 minutes timeout
            wait_time = 0
            while wait_time < max_wait_time:
                if (
                    os.path.exists(output_filepath)
                    and os.path.getsize(output_filepath) > 0
                ):
                    self.logger.info(f"Dataset for {self.bac_name} is now available.")
                    return
                time.sleep(10)
                wait_time += 10
            self.logger.warning(
                f"Timeout waiting for dataset generation for {self.bac_name}."
            )
            return

        try:
            # Continue with dataset generation
            self.logger.info(f"Generating dataset for {self.bac_name}...")

            if not file_list:
                self.logger.warning(f"No feature files found in {self.feature_dir}")

            # Check for cached dataset
            cache_file = os.path.join(self.root_dir, self.cache_dir, f"{self.bac_name}_dataset.csv")
            if os.path.exists(cache_file):
                self.logger.info(f"Using cached dataset for: {self.bac_name}")
                df = pd.read_csv(cache_file)
                self._save_dataset(df)
                return

            # Validate file list
            if not file_list:
                self.logger.error(
                    f"No feature files found in {self.feature_dir} for {self.bac_name}"
                )
                # Create an empty DataFrame with the correct columns to avoid errors
                columns = [i for i in self._generate_kmer_combinations()]
                columns.extend(self.antibiotic_list)
                self._save_dataset(pd.DataFrame(columns=columns))
                return

            for file_name in file_list:
                self.logger.info(f"Processing sequence: {file_name}")
                sra_id = file_name.replace(".csv", "")

                kmer_list = self._get_kmer_counts(file_name=file_name)
                # mic_list = self._get_mic_data(seq_name=sra_id)
                phenotype_list = self._get_phenotype_data(seq_name=sra_id)

                list_data = [*kmer_list, *phenotype_list]
                seq_data.append(list_data)

            columns = [i for i in self._generate_kmer_combinations()]
            columns.extend(self.antibiotic_list)
            self.logger.debug(f"Generated {len(columns)} columns")
            df = pd.DataFrame(seq_data, columns=columns)

            # Cache the generated dataset
            df.to_csv(os.path.join(self.root_dir, cache_file), index=False)
            self._save_dataset(df)

        finally:
            self._release_lock(lock_fd)

    def _save_dataset(self, df: pd.DataFrame) -> None:
        """
        Saves the generated dataset to a CSV file.

        Args:
            df: A DataFrame representing the generated dataset.
        """
        output_filepath = os.path.join(self.root_dir, self.output_dir, f"{self.bac_name}_dataset.csv")
        df.to_csv(output_filepath, index=False)

        # Validate the saved dataset
        if df.empty:
            self.logger.warning(f"Generated dataset is empty for {self.bac_name}")
        else:
            self.logger.info(
                f"Dataset with {len(df)} rows and {len(df.columns)} columns saved to: {output_filepath}"
            )
            self.logger.debug(f"First few columns: {', '.join(list(df.columns)[:5])}")
