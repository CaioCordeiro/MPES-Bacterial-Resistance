import os
import pandas as pd
import numpy as np
# Assuming you have a logging utility similar to other parts of your project
# If not, you can use the standard logging module or remove this
from utils.logging_config import get_logger # Or your project's logger setup

class ProteinDataset:
    """
    A class to load, process, and represent a protein dataset from a CSV file.
    """

    def __init__(
        self,
        root_dir: str = '', # Base directory for data
        data_subdir: str = "raw-data/Ac_Sa_Ca_KL_Ec", # Subdirectory for raw data
        input_csv_filename: str = "aac_all.csv",
        column_to_drop: str = "Feature",
        output_subdir: str = "processed_protein_data",
        name: str = "ProteinDataset"
    ):
        """
        Initializes the ProteinDataset.

        Args:
            root_dir: The root directory where data subdirectories are located.
            data_subdir: Subdirectory under root_dir containing the input CSV.
            input_csv_filename: Name of the input CSV file.
            column_to_drop: Name of the column to drop from the CSV.
            output_subdir: Subdirectory under root_dir to save processed data.
            name: Name of the dataset.
        """
        self.logger = get_logger() # Initialize logger
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, data_subdir)
        self.input_csv_path = os.path.join(self.data_dir, input_csv_filename)
        self.column_to_drop = column_to_drop
        
        self.output_dir = os.path.join(self.root_dir, output_subdir)
        self.processed_csv_filename = f"processed_{input_csv_filename}"
        self.processed_csv_output_path = os.path.join(self.output_dir, self.processed_csv_filename)
        
        self._name = name
        self._treated_data: pd.DataFrame = None

        self._ensure_dir(self.output_dir)

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self._name

    def _ensure_dir(self, directory_path: str):
        """Ensures that a directory exists, creating it if necessary."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            self.logger.info(f"Created directory: {directory_path}")

    def _load_and_process_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file, drops the specified column, and stores it.
        """
        self.logger.info(f"Attempting to load CSV from: {self.input_csv_path}")
        try:
            df = pd.read_csv(self.input_csv_path)
            self.logger.info(f"Successfully loaded {self.input_csv_path}")
            self.logger.debug(f"Initial columns: {df.columns.tolist()}")
            self.logger.debug(f"DataFrame shape: {df.shape}")

            if self.column_to_drop in df.columns:
                df_processed = df.drop(columns=[self.column_to_drop])
                self.logger.info(f"Dropped column: '{self.column_to_drop}'")
                self.logger.debug(f"Columns after dropping: {df_processed.columns.tolist()}")
            else:
                self.logger.warning(f"Column '{self.column_to_drop}' not found in the CSV at {self.input_csv_path}.")
                self.logger.warning(f"Available columns are: {df.columns.tolist()}")
                self.logger.warning("Please ensure the column name matches exactly, including case.")
                self.logger.warning("No columns were dropped. Using original DataFrame.")
                df_processed = df.copy()

            self._treated_data = df_processed
            return self._treated_data
            
        except FileNotFoundError:
            self.logger.error(f"Error: File not found at {self.input_csv_path}")
            self._treated_data = pd.DataFrame() # Return empty DataFrame
            return self._treated_data
        except pd.errors.EmptyDataError:
            self.logger.error(f"Error: The file at {self.input_csv_path} is empty.")
            self._treated_data = pd.DataFrame()
            return self._treated_data
        except Exception as e:
            self.logger.error(f"Error processing CSV file {self.input_csv_path}: {e}", exc_info=True)
            self._treated_data = pd.DataFrame()
            return self._treated_data

    @property
    def treated_data(self) -> pd.DataFrame:
        """
        Returns the treated protein dataset as a Pandas DataFrame.
        Loads and processes the data if it hasn't been already.
        """
        if self._treated_data is None:
            self.logger.info("Treated data not yet loaded. Processing now...")
            self._load_and_process_data()
        return self._treated_data

    def save_processed_data(self, output_path: str = None) -> None:
        """
        Saves the processed DataFrame to a CSV file.

        Args:
            output_path: Optional path to save the file. Defaults to pre-configured path.
        """
        if self._treated_data is None or self._treated_data.empty:
            self.logger.warning("No treated data available to save.")
            return

        save_path = output_path if output_path else self.processed_csv_output_path
        
        self.logger.info(f"Saving processed data to: {save_path}")
        try:
            self._treated_data.to_csv(save_path, index=False)
            self.logger.info(f"Processed CSV file saved successfully to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed CSV to {save_path}: {e}", exc_info=True)

    def get_summary(self) -> None:
        """Generates and prints summary statistics for the processed DataFrame."""
        if self._treated_data is None or self._treated_data.empty:
            self.logger.info("No processed data available to summarize.")
            return

        self.logger.info("--- Protein Data Summary ---")
        self.logger.info(f"Dataset Name: {self.name}")
        self.logger.info(f"Source File: {self.input_csv_path}")
        self.logger.info(f"Shape (rows, columns): {self._treated_data.shape}")
        
        self.logger.info("First 5 rows of processed data:")
        # Pandas info and describe print to stdout, so using logger.info for them might be verbose
        # or require capturing stdout. For simplicity, direct print or selective logging.
        print(self._treated_data.head()) 
        
        self.logger.info("Data types and non-null values (DataFrame.info()):")
        # self._treated_data.info() # This prints to stdout
        # For logging, you might capture it:
        # import io
        # buffer = io.StringIO()
        # self._treated_data.info(buf=buffer)
        # self.logger.info(buffer.getvalue())
        
        numerical_df = self._treated_data.select_dtypes(include=np.number)
        if not numerical_df.empty:
            self.logger.info("Descriptive statistics for numerical columns:")
            print(numerical_df.describe())
        else:
            self.logger.info("No numerical columns to describe.")
        self.logger.info("-----------------------------")

    def run_pipeline(self) -> pd.DataFrame:
        """
        Ensures the data is loaded and processed, then returns it.
        This is the main method to get the final dataset.
        """
        self.logger.info(f"Running pipeline for {self.name}...")
        data = self.treated_data # Accessing property ensures loading
        if not data.empty:
            self.logger.info(f"Pipeline for {self.name} completed successfully.")
        else:
            self.logger.warning(f"Pipeline for {self.name} resulted in empty data.")
        return data