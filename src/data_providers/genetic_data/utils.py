import csv
import os

import pandas as pd


def create_folder(path: str):
    """
    Create folder if not exists.

    Args:
        path: string, name of folder to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def write_csv_file(fname, data, header=None, *args, **kwargs):
    """
    Write data to file.

    Args:
        fname: string, name of file to write
        data: list of list of items
    """
    csv_file = csv.writer(open(fname, "w"), *args, **kwargs)

    if header:
        csv_file.writerow(header)

    for row in data:
        csv_file.writerow(row)


def normalize_mic(value: str):
    """
    Normalize value following Nguyen rules.

    Args:
        fname: string, value to normalize
    """
    if "/" in value:
        value = value.split("/")[0]

    if "<=" in value or ">=" in value:
        return float(value.replace("<=", "").replace(">=", ""))

    if "<" in value:
        return float(value.replace("<", "")) / 2

    if ">" in value:
        return float(value.replace(">", "")) * 2

    return float(value)


def check_dataset_file(filepath: str, verbose: bool = True):
    """
    Check if a dataset file exists and contains valid data.

    Args:
        filepath: Path to the dataset file
        verbose: Whether to print detailed information

    Returns:
        bool: True if file exists and contains data, False otherwise
    """
    if not os.path.exists(filepath):
        if verbose:
            print(f"Dataset file not found: {filepath}")
        return False

    if os.path.getsize(filepath) <= 1:
        if verbose:
            print(f"Dataset file is empty: {filepath}")
        return False

    return True
