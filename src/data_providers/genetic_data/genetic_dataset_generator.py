import csv
import os
from datetime import datetime
from itertools import product
from typing import Dict, Generator, List

import pandas as pd

from constants.constants import (ANTIBIOTIC_FILE, ANTIBIOTIC_LIST,
                                 DATASET_OUTPUT_DIR, FEATURE_DIR, K_SIZE)

from .utils import create_folder  # Assuming these are in 'utils.py'
from .utils import normalize_mic, write_csv_file


class DatasetGenerator:
    """
    A class to generate a dataset by combining k-mer counts from feature files
    with antibiotic Minimum Inhibitory Concentration (MIC) data.
    """

    def __init__(
        self,
        feature_dir: str = FEATURE_DIR,
        output_dir: str = DATASET_OUTPUT_DIR,
        antibiotic_file: str = ANTIBIOTIC_FILE,
        antibiotic_list: List[str] = None,
        k_size: int = K_SIZE,
        bac_name: int = "kleb",
    ):
        """
        Initializes the DatasetGenerator.

        Args:
            feature_dir: The directory containing the k-mer count CSV files.
            output_dir: The directory where the generated dataset CSV will be saved.
            antibiotic_file: The path to the CSV file containing antibiotic MIC data.
            antibiotic_list: A list of antibiotics to include in the dataset.
                             Defaults to the list defined in constants.py.
            k_size: The size of the k-mers used to generate the feature files.
        """
        self.bac_name = bac_name
        self.feature_dir = feature_dir
        self.output_dir = output_dir
        self.antibiotic_file = (
            f"{antibiotic_file}/{self.bac_name}/antibotic_relation.csv"
        )
        self.antibiotic_list = (
            antibiotic_list if antibiotic_list is not None else ANTIBIOTIC_LIST
        )
        self.k_size = k_size
        create_folder(self.output_dir)

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
            print(f"MIC data for {seq_name}: {mic_list}")
            return mic_list
        except FileNotFoundError:
            print(
                f"Error: Antibiotic relation file not found at {self.antibiotic_file}"
            )
            return [0.0] * len(self.antibiotic_list)
        except KeyError as e:
            print(f"Error: Column '{e}' not found in {self.antibiotic_file}")
            return [0.0] * len(self.antibiotic_list)

    def _get_kmer_counts(self, file_name: str) -> List[int]:
        """
        Reads k-mer counts from a CSV file.

        Args:
            file_name: The name of the CSV file containing k-mer counts.

        Returns:
            A list of k-mer counts.
        """
        filepath = os.path.join(self.feature_dir, file_name)
        try:
            with open(filepath, "r") as csvfile:
                csvreader = csv.reader(csvfile)
                row = [int(item) for item in next(csvreader)]
                return row
        except FileNotFoundError:
            print(f"Warning: Feature file not found: {filepath}")
            return []
        except StopIteration:
            print(f"Warning: Feature file is empty: {filepath}")
            return []
        except Exception as e:
            print(f"Error reading feature file {filepath}: {e}")
            return []

    def generate_dataset(self) -> None:
        """
        Generates the combined dataset by iterating through feature files,
        retrieving k-mer counts and MIC data, and saving the result to a CSV file.
        """
        file_list = os.listdir(self.feature_dir)
        seq_data: List[List] = []
        print(file_list)
        for file_name in file_list:
            print("RUNNING FOR SEQ: ", file_name)
            sra_id = file_name.replace(".csv", "")

            kmer_list = self._get_kmer_counts(file_name=file_name)
            mic_list = self._get_mic_data(seq_name=sra_id)

            list_data = [*kmer_list, *mic_list]
            seq_data.append(list_data)

        columns = [i for i in self._generate_kmer_combinations()]
        columns.extend(self.antibiotic_list)
        print("Generated columns:", len(columns))
        self._save_dataset(seq_data=seq_data, columns=columns)

    def _save_dataset(self, seq_data: List[List], columns: List[str]) -> None:
        """
        Saves the generated dataset to a CSV file.

        Args:
            seq_data: A list of lists, where each inner list represents a sequence's data.
            columns: A list of column names for the CSV file.
        """
        output_filepath = os.path.join(
            self.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        write_csv_file(output_filepath, seq_data, header=columns)
        print(f"Dataset saved to: {output_filepath}")
