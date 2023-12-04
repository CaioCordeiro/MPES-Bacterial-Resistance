import os
import csv
import pandas as pd
from core.utils import create_folder, write_csv_file, normalize_mic
from datetime import datetime
from config import Config

FILE_DIR = 'data/features'
OUTPUT_DIR = 'data/datasets'
ANTIBIOTIC_FILE = 'data/antibotic_relation.csv'


def main():
    file_list = os.listdir(FILE_DIR)
    seq_data = []

    for file_name in file_list:
        print("RUNNING FOR SEQ: ", file_name)
        sra_id = file_name.replace(".csv", "")

        kmer_list = get_kmers_count_data(file_name=file_name)
        mic_list = get_data_from_antibiotic_file(seq_name=sra_id)

        list_data = [*kmer_list, *mic_list]
        seq_data.append(list_data)

    generate_dataset(seq_data=seq_data)


def get_data_from_antibiotic_file(seq_name: str):
    # OPEN CSV FILE
    csv = pd.read_csv(ANTIBIOTIC_FILE,  header=0, names=[
                      "genome", "sra_id", "patric_id", "antibiotic", "mic_actual", "mic_predicted"])

    # SEARCH LINES FOR SEQ_NAME
    sra_relation = csv.loc[csv['sra_id'] == seq_name][[
        "antibiotic", "mic_actual"]]
    sra_relation = sra_relation.set_index("antibiotic")

    # CREATE DICT (ANTIBIOTIC, MIC)
    actual_mic = sra_relation.to_dict()["mic_actual"]

    # GET ORDERED MIC
    mic_list = [normalize_mic(actual_mic[antibiotic]) if antibiotic in actual_mic else 0.0
                for antibiotic in Config.ANTIBIOTIC_LIST]

    return mic_list


def get_kmers_count_data(file_name: str):
    csvreader = csv.reader(open(FILE_DIR+"/"+file_name))
    row = [int(item) for item in next(csvreader)]

    return row


def generate_dataset(seq_data: list):

    create_folder(OUTPUT_DIR)
    write_csv_file(
        str(OUTPUT_DIR + "/" + str(datetime.now()) + ".csv"), seq_data)


if __name__ == "__main__":
    main()
