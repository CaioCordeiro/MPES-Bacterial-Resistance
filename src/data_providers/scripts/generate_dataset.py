import csv
import os
from datetime import datetime
from itertools import product

import pandas as pd
from utils import create_folder, normalize_mic, write_csv_file

ANTIBIOTIC_LIST = [
    "Amikacin",
    "Ampicillin",
    "Ampicillin/Sulbactam",
    "Aztreonam",
    "Cefazolin",
    "Cefepime",
    "Cefoxitin",
    "Ceftazidime",
    "Ceftriaxone",
    "Cefuroxime sodium",
    "Ciprofloxacin",
    "Gentamicin",
    "Imipenem",
    "Levofloxacin",
    "Meropenem",
    "Nitrofurantoin",
    "Piperacillin/Tazobactam",
    "Tetracycline",
    "Tobramycin",
    "Trimethoprim/Sulfamethoxazole",
    "Vancomycin",
]


FILE_DIR = "data/features"
OUTPUT_DIR = "data/datasets"
ANTIBIOTIC_FILE = "data/antibotic_relation.csv"


def main():
    file_list = os.listdir(FILE_DIR)
    seq_data = []
    columns_mic = [
        "genome",
        "sra_id",
        "patric_id",
        "antibiotic",
        "mic_actual",
        "mic_predicted",
    ]
    for file_name in file_list:
        print("RUNNING FOR SEQ: ", file_name)
        sra_id = file_name.replace(".csv", "")

        kmer_list = get_kmers_count_data(file_name=file_name)
        mic_list = get_data_from_antibiotic_file(seq_name=sra_id, columns=columns_mic)

        list_data = [*kmer_list, *mic_list]
        seq_data.append(list_data)
    columns = [i for i in generate_kmer_combinations(5)]
    columns.extend(ANTIBIOTIC_LIST)
    print(columns)
    generate_dataset(seq_data=seq_data, columns=columns)


def generate_kmer_combinations(n, alphabet="ACGT"):
    """
    Generates all possible k-mer combinations of length n in reverse lexicographical order.

    Args:
        n: The length of the k-mer (integer).
        alphabet: The set of characters to use in the k-mers (string).
                Defaults to "ACGT".

    Yields:
        str: The next k-mer combination in reverse order.
    """
    for combination_tuple in list(product(alphabet, repeat=n)):
        yield "".join(combination_tuple)


def get_data_from_antibiotic_file(
    seq_name: str,
    columns=[
        "genome",
        "sra_id",
        "patric_id",
        "antibiotic",
        "mic_actual",
        "mic_predicted",
    ],
):
    # OPEN CSV FILE
    csv = pd.read_csv(ANTIBIOTIC_FILE, header=0, names=columns)

    # SEARCH LINES FOR SEQ_NAME
    sra_relation = csv.loc[csv["sra_id"] == seq_name][["antibiotic", "mic_actual"]]
    sra_relation = sra_relation.set_index("antibiotic")

    # CREATE DICT (ANTIBIOTIC, MIC)
    actual_mic = sra_relation.to_dict()["mic_actual"]

    # GET ORDERED MIC
    mic_list = [
        normalize_mic(actual_mic[antibiotic]) if antibiotic in actual_mic else 0.0
        for antibiotic in ANTIBIOTIC_LIST
    ]
    print(mic_list)
    return mic_list


def get_kmers_count_data(file_name: str):
    csvreader = csv.reader(open(FILE_DIR + "/" + file_name))
    row = [int(item) for item in next(csvreader)]

    return row


def generate_dataset(seq_data: list, columns: list):
    create_folder(OUTPUT_DIR)
    write_csv_file(
        str(OUTPUT_DIR + "/" + str(datetime.now()) + ".csv"), seq_data, columns
    )


if __name__ == "__main__":
    main()
