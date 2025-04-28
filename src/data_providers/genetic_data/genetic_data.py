import os
import sys
import warnings
from typing import List, Optional

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

import constants.constants as const  # Correct import path
from data_providers.genetic_data.dataset_cache_manager import \
    DatasetCacheManager
from data_providers.genetic_data.genetic_data_downloader import SraDownloader
from data_providers.genetic_data.genetic_dataset_generator import \
    DatasetGenerator  # Assuming this class generates the final dataset DataFrame
from data_providers.genetic_data.kmer_extrator import KmerExtractor
from data_providers.genetic_data.utils import \
    create_folder  # Assuming create_folder is in utils.py
from entities.interfaces.dataset import DatasetInterface
from utils.logging_config import get_logger


class GeneticDataset(DatasetInterface):
    """
    A class to fetch, process, and represent a genetic dataset, adhering to the DatasetInterface.
    """

    def __init__(
        self,
        sra_ids: Optional[List[str]] = None,
        exclude_ids: List[str] = const.EXCLUDE_LIST,
        raw_data_output: str = const.FILE_DIR,
        kmer_size: int = const.K_SIZE,
        feature_output: str = const.FEATURE_DIR,
        dataset_output: str = const.DATASET_OUTPUT_DIR,
        name: str = "GeneticDataset",
        max_sra_ids: Optional[int] = None,
        metric_provider=None,
        bac_name: str = "kleb",
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
            max_sra_ids: Maximum number of SRA IDs to process. If None, all IDs are processed.
            metric_provider: Optional object to provide metrics.
        """
        super().__init__(raw_data=None, name=name, metric_provider=metric_provider)
        self.bac_name = bac_name
        self.logger = get_logger()
        self.max_sra_ids = max_sra_ids
        self.sra_ids = (
            sra_ids if sra_ids is not None else self._get_sra_ids_from_relation_file()
        )
        self.exclude_ids = exclude_ids
        self.raw_data_output_dir = raw_data_output + "/" + self.bac_name
        self.kmer_size = kmer_size
        self.feature_output_dir = feature_output + "/" + self.bac_name
        self.dataset_output_dir = dataset_output + "/" + self.bac_name
        self._treated_data: DataFrame = None
        self.cache_manager = DatasetCacheManager(
            cache_dir=const.CACHE_DIR, bac_name=self.bac_name
        )
        create_folder(self.raw_data_output_dir)
        create_folder(self.feature_output_dir)
        create_folder(self.dataset_output_dir)

    def _get_sra_ids_from_relation_file(self) -> List[str]:
        """
        Reads SRA IDs from the antibiotic relation file for the current bacteria.

        Returns:
            A list of SRA IDs.
        """
        relation_file = f"data/anti_rel/{self.bac_name}/antibotic_relation.csv"
        if not os.path.exists(relation_file):
            self.logger.warning(
                f"Antibiotic relation file not found at {relation_file}"
            )
            return getattr(const, f"SRA_ID_LIST_{self.bac_name.upper()}", [])

        try:
            df = pd.read_csv(relation_file)
            ids = df["SRA ID"].unique().tolist()[: self.max_sra_ids]
            return ids
        except Exception as e:
            self.logger.error(f"Error reading antibiotic relation file: {e}")
            return getattr(const, f"SRA_ID_LIST_{self.bac_name.upper()}", [])

    def _fetch_raw_data(self) -> None:
        """
        Fetches raw genetic data (FASTQ files) from the NCBI SRA database.
        """
        # Check if we already have enough feature files
        feature_files = [
            f for f in os.listdir(self.feature_output_dir) if f.endswith(".csv")
        ]
        if len(feature_files) >= (self.max_sra_ids or float("inf")):
            self.logger.debug(
                f"Sufficient feature files ({len(feature_files)}) already exist. Skipping download."
            )
            return

        # Check if dataset already exists in cache
        if self.cache_manager.dataset_exists():
            return
        self.logger.info("Fetching raw genetic data from SRA...")
        downloader = SraDownloader(
            sra_id_list=self.sra_ids,
            exclude_list=self.exclude_ids,
            bac_name=self.bac_name,
            max_sra_ids=self.max_sra_ids,
        )
        downloader.download()
        self.logger.info("Raw genetic data fetching finished.")

    def _extract_features(self) -> None:
        """
        Extracts k-mer features from the fetched raw genetic data and saves them to CSV files.
        """
        # Check if we already have enough feature files
        feature_files = [
            f for f in os.listdir(self.feature_output_dir) if f.endswith(".csv")
        ]
        if len(feature_files) >= (self.max_sra_ids or float("inf")):
            self.logger.debug(
                f"Sufficient feature files ({len(feature_files)}) already exist. Skipping k-mer extraction."
            )
            return

        # Check if dataset already exists in cache
        if self.cache_manager.dataset_exists():
            return
        self.logger.info("Extracting k-mer features...")
        extractor = KmerExtractor(k_size=self.kmer_size, bac_name=self.bac_name)
        extractor.process_sequences()
        self.logger.info("K-mer feature extraction finished.")

    def _generate_combined_dataset(self) -> DataFrame:
        """
        Generates the combined dataset DataFrame by reading feature files and MIC data.
        """
        # Try to load from cache first
        cached_df = self.cache_manager.load_dataset()
        if cached_df is not None:
            self.logger.info(f"Loaded dataset from cache for {self.bac_name}")
            self._treated_data = cached_df
            return self._treated_data

        self.logger.info("Generating the combined dataset DataFrame...")
        generator = DatasetGenerator(
            feature_dir=self.feature_output_dir,
            k_size=self.kmer_size,
            output_dir=self.dataset_output_dir,
            bac_name=self.bac_name,
        )
        # Modify DatasetGenerator to return the DataFrame instead of saving to a file
        # This might require changes in the DatasetGenerator class.
        # For now, assuming it saves and we load it.
        generator.generate_dataset()  # This will save the CSV
        list_of_files = [
            os.path.join(self.dataset_output_dir, f)
            for f in os.listdir(self.dataset_output_dir)
            if f.endswith(".csv")
        ]
        if not list_of_files:
            raise FileNotFoundError(
                f"No dataset CSV file found in {self.dataset_output_dir}"
            )

        latest_file = max(list_of_files, key=os.path.getctime)

        # Check if file exists and is not empty
        if not os.path.exists(latest_file) or os.path.getsize(latest_file) <= 1:
            raise ValueError(f"Dataset file {latest_file} is empty or does not exist")

        # Load the dataset
        try:
            df = pd.read_csv(latest_file)
        except pd.errors.EmptyDataError:
            # If the file is empty but exists, create a new dataset
            self.logger.warning(f"Dataset file {latest_file} is empty. Regenerating...")
            generator = DatasetGenerator(
                feature_dir=self.feature_output_dir,
                k_size=self.kmer_size,
                output_dir=self.dataset_output_dir,
                bac_name=self.bac_name,
            )
            generator.generate_dataset()
            df = pd.read_csv(latest_file)
        if df.empty:
            raise ValueError(f"Dataset loaded from {latest_file} is empty")

        self._treated_data = DataFrame(df)

        # Cache the dataset for future use
        self.cache_manager.save_dataset(self._treated_data)
        self.logger.info("Combined dataset DataFrame generated.")
        return self._treated_data

    @property
    def treated_data(self) -> DataFrame:
        """
        Returns the treated genetic dataset as a Pandas DataFrame.
        Fetches, extracts, and generates the dataset if it hasn't been created yet.
        """
        if self._treated_data is None:
            # Try to load from cache first
            cached_df = self.cache_manager.load_dataset()
            if cached_df is not None:
                self.logger.info(f"Loaded dataset from cache for {self.bac_name}")
                self._treated_data = cached_df
            else:
                self._fetch_raw_data()
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
