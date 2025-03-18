import os
import sys
from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from constants import constants as const  # Correct import path
from data_providers.genetic_data.genetic_data_downloader import SraDownloader
from data_providers.genetic_data.genetic_dataset_generator import \
    DatasetGenerator  # Assuming this class generates the final dataset DataFrame
from data_providers.genetic_data.kmer_extrator import KmerExtractor
from data_providers.genetic_data.utils import \
    create_folder  # Assuming create_folder is in utils.py
from entities.interfaces.dataset import DatasetInterface


class GeneticDataset(DatasetInterface):
    """
    A class to fetch, process, and represent a genetic dataset, adhering to the DatasetInterface.
    """

    def __init__(
        self,
        sra_ids: List[str] = const.SRA_ID_LIST,
        exclude_ids: List[str] = const.EXCLUDE_LIST,
        raw_data_output: str = const.FILE_DIR,
        kmer_size: int = const.K_SIZE,
        feature_output: str = const.FEATURE_DIR,
        dataset_output: str = const.DATASET_OUTPUT_DIR,
        name: str = "GeneticDataset",
        metric_provider=None,
    ):
        """
        Initializes the GeneticDataset.

        Args:
            sra_ids: List of SRA IDs to process.
            exclude_ids: List of SRA IDs to exclude.
            raw_data_output: Directory to save raw genetic data.
            kmer_size: Size of k-mers to extract as features.
            feature_output: Directory to save k-mer features.
            dataset_output: Directory to save the final dataset.
            name: Name of the dataset.
            metric_provider: Optional object to provide metrics.
        """
        super().__init__(raw_data=None, name=name, metric_provider=metric_provider)
        self.sra_ids = sra_ids
        self.exclude_ids = exclude_ids
        self.raw_data_output_dir = raw_data_output
        self.kmer_size = kmer_size
        self.feature_output_dir = feature_output
        self.dataset_output_dir = dataset_output
        self._treated_data: DataFrame = None
        create_folder(self.raw_data_output_dir)
        create_folder(self.feature_output_dir)
        create_folder(self.dataset_output_dir)

    def _fetch_raw_data(self) -> None:
        """
        Fetches raw genetic data (FASTQ files) from the NCBI SRA database.
        """
        print("Fetching raw genetic data from SRA...")
        downloader = SraDownloader(
            sra_id_list=self.sra_ids,
            exclude_list=self.exclude_ids,
            output_dir=self.raw_data_output_dir,
        )
        downloader.download()
        print("Raw genetic data fetching finished.")

    def _extract_features(self) -> None:
        """
        Extracts k-mer features from the fetched raw genetic data and saves them to CSV files.
        """
        print("Extracting k-mer features...")
        extractor = KmerExtractor(
            k_size=self.kmer_size,
            file_dir=self.raw_data_output_dir,
            output_dir=self.feature_output_dir,
        )
        extractor.process_sequences()
        print("K-mer feature extraction finished.")

    def _generate_combined_dataset(self) -> DataFrame:
        """
        Generates the combined dataset DataFrame by reading feature files and MIC data.
        """
        print("Generating the combined dataset DataFrame...")
        generator = DatasetGenerator(
            feature_dir=self.feature_output_dir,
            k_size=self.kmer_size,
            output_dir=self.dataset_output_dir,
        )
        # Modify DatasetGenerator to return the DataFrame instead of saving to a file
        # This might require changes in the DatasetGenerator class.
        # For now, assuming it saves and we load it.
        generator.generate_dataset()  # This will save the CSV
        list_of_files = [
            f for f in os.listdir(self.dataset_output_dir) if f.endswith(".csv")
        ]
        if not list_of_files:
            raise FileNotFoundError(
                f"No dataset CSV file found in {self.dataset_output_dir}"
            )
        latest_file = max(
            [os.path.join(self.dataset_output_dir, f) for f in list_of_files],
            key=os.path.getctime,
        )
        self._treated_data = DataFrame(pd.read_csv(latest_file))
        print("Combined dataset DataFrame generated.")
        return self._treated_data

    @property
    def treated_data(self) -> DataFrame:
        """
        Returns the treated genetic dataset as a Pandas DataFrame.
        Fetches, extracts, and generates the dataset if it hasn't been created yet.
        """
        if self._treated_data is None:
            # self._fetch_raw_data()
            self._extract_features()
            self._treated_data = self._generate_combined_dataset()
        return self._treated_data

    def splitted_dataset(
        self, test_size: float, random_state: int = None
    ) -> list[DataFrame]:
        """
        Splits the treated dataset into training and testing sets.

        Args:
            test_size: The proportion of the dataset to use for the test set (e.g., 0.2 for 20%).
            random_state: Optional random seed for reproducibility.

        Returns:
            A list containing two DataFrames: [train_df, test_df].
        """
        if self.treated_data is None:
            raise ValueError(
                "Treated data is not available. Call treated_data property first."
            )

        train_df, test_df = train_test_split(
            self.treated_data, test_size=test_size, random_state=random_state
        )
        return [train_df, test_df]

    @property
    def metrics(self):
        """
        Returns the metrics provider object (if set).
        """
        return self.metric_provider

    def run_pipeline(self) -> DataFrame:
        """
        Executes the entire data processing pipeline and returns the treated dataset.
        """
        return self.treated_data
