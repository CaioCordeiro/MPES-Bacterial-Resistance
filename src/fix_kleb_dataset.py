#!/usr/bin/env python3
"""
Script to manually fix the Klebsiella dataset.
This can be run independently to troubleshoot dataset generation issues.
"""

import os
import sys
from itertools import product

import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from constants import constants as const
from data_providers.genetic_data.utils import create_folder, normalize_mic


def generate_kmer_combinations(n=5, alphabet="ACGT"):
    """Generates all possible k-mer combinations of length n."""
    return ["".join(combo) for combo in product(alphabet, repeat=n)]


def get_mic_data(sra_id, antibiotic_file, antibiotic_list):
    """Get MIC data for a specific SRA ID."""
    try:
        # Read the CSV file
        csv_data = pd.read_csv(antibiotic_file)

        # Filter rows for the specific SRA ID
        sra_rows = csv_data[csv_data["SRA ID"] == sra_id]

        # Create a dictionary mapping antibiotics to their MIC values
        mic_dict = {}
        for _, row in sra_rows.iterrows():
            antibiotic = row["Antibiotic"]
            mic_value = row["Actual MIC"]
            mic_dict[antibiotic] = normalize_mic(mic_value)

        # Return MIC values for each antibiotic in the list
        return [mic_dict.get(antibiotic, 0.0) for antibiotic in antibiotic_list]
    except Exception as e:
        print(f"Error getting MIC data for {sra_id}: {e}")
        return [0.0] * len(antibiotic_list)


def get_kmer_counts(file_path):
    """Read k-mer counts from a CSV file."""
    try:
        with open(file_path, "r") as csvfile:
            import csv

            csvreader = csv.reader(csvfile)
            row = [int(item) for item in next(csvreader)]
            return row
    except Exception as e:
        print(f"Error reading feature file {file_path}: {e}")
        return []


def main():
    """Main function to fix the Klebsiella dataset."""
    bac_name = "kleb"

    # Define paths
    feature_dir = f"{const.FEATURE_DIR}/{bac_name}"
    dataset_dir = f"{const.DATASET_OUTPUT_DIR}/{bac_name}"
    antibiotic_file = f"{const.ANTIBIOTIC_FILE}/{bac_name}/antibotic_relation.csv"
    output_file = f"{dataset_dir}/{bac_name}_dataset.csv"

    # Create directories if they don't exist
    for directory in [feature_dir, dataset_dir]:
        create_folder(directory)

    # Check if feature files exist
    if not os.path.exists(feature_dir) or not os.listdir(feature_dir):
        print(
            f"No feature files found in {feature_dir}. Please run the k-mer extraction first."
        )
        return

    # Get list of feature files
    file_list = [f for f in os.listdir(feature_dir) if f.endswith(".csv")]
    if not file_list:
        print(f"No CSV files found in {feature_dir}")
        return

    print(f"Found {len(file_list)} feature files.")

    # Generate dataset
    seq_data = []
    for file_name in file_list:
        print(f"Processing {file_name}...")
        sra_id = file_name.replace(".csv", "")

        # Get k-mer counts and MIC data
        kmer_list = get_kmer_counts(os.path.join(feature_dir, file_name))
        if not kmer_list:
            print(f"Warning: Empty or invalid k-mer data for {sra_id}")
            continue

        mic_list = get_mic_data(sra_id, antibiotic_file, const.ANTIBIOTIC_LIST)

        # Combine data
        list_data = [*kmer_list, *mic_list]
        seq_data.append(list_data)

    if not seq_data:
        print("No valid data collected. Dataset cannot be created.")
        return

    # Generate column names
    columns = generate_kmer_combinations(const.K_SIZE)
    columns.extend(const.ANTIBIOTIC_LIST)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(seq_data, columns=columns)
    df.to_csv(output_file, index=False)

    print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Saved to {output_file}")

    # Verify the dataset
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print("Dataset verification: SUCCESS")
    else:
        print("Dataset verification: FAILED")


if __name__ == "__main__":
    main()
