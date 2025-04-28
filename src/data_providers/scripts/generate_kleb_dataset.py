#!/usr/bin/env python3
"""
Script to manually generate the Klebsiella dataset.
This can be run independently to troubleshoot dataset generation issues.
"""

import os
import sys

import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.constants import constants as const
from src.data_providers.genetic_data.genetic_data_downloader import \
    SraDownloader
from src.data_providers.genetic_data.genetic_dataset_generator import \
    DatasetGenerator
from src.data_providers.genetic_data.kmer_extrator import KmerExtractor
from src.data_providers.genetic_data.utils import (check_dataset_file,
                                                   create_folder)


def main():
    """Main function to generate the Klebsiella dataset."""
    bac_name = "kleb"

    # Create necessary directories
    raw_data_dir = f"{const.FILE_DIR}/{bac_name}"
    feature_dir = f"{const.FEATURE_DIR}/{bac_name}"
    dataset_dir = f"{const.DATASET_OUTPUT_DIR}/{bac_name}"

    for directory in [raw_data_dir, feature_dir, dataset_dir]:
        create_folder(directory)

    # Step 1: Download data if needed
    sra_ids = const.SRA_ID_LIST_KLEB
    print(f"SRA IDs for Klebsiella: {sra_ids}")

    if not sra_ids:
        print("Error: No SRA IDs defined for Klebsiella. Please check constants.py")
        return

    # Check if raw data files exist
    raw_data_exists = all(
        os.path.exists(f"{raw_data_dir}/{sra_id}/{sra_id}.fasta") for sra_id in sra_ids
    )

    if not raw_data_exists:
        print("Downloading raw data files...")
        downloader = SraDownloader(sra_id_list=sra_ids, bac_name=bac_name)
        downloader.download()
    else:
        print("Raw data files already exist. Skipping download.")

    # Step 2: Extract k-mers if needed
    feature_files_exist = all(
        os.path.exists(f"{feature_dir}/{sra_id}.csv") for sra_id in sra_ids
    )

    if not feature_files_exist:
        print("Extracting k-mers...")
        extractor = KmerExtractor(k_size=const.K_SIZE, bac_name=bac_name)
        extractor.process_sequences()
    else:
        print("Feature files already exist. Skipping k-mer extraction.")

    # Step 3: Generate dataset
    dataset_file = f"{dataset_dir}/{bac_name}_dataset.csv"

    # Force regeneration of dataset
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
        print(f"Removed existing dataset file: {dataset_file}")

    print("Generating dataset...")
    generator = DatasetGenerator(
        feature_dir=feature_dir,
        output_dir=dataset_dir,
        k_size=const.K_SIZE,
        bac_name=bac_name,
    )
    generator.generate_dataset()

    # Verify the generated dataset
    if check_dataset_file(dataset_file):
        df = pd.read_csv(dataset_file)
        print(
            f"Dataset generated successfully with {len(df)} rows and {len(df.columns)} columns"
        )
        print(f"First few columns: {', '.join(list(df.columns)[:5])}")
    else:
        print("Failed to generate a valid dataset")


if __name__ == "__main__":
    main()
