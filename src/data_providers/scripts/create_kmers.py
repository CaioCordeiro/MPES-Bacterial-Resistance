import os
import time

from utils import create_folder, write_csv_file

K_SIZE = 5

FILE_DIR = "data/raw_data"
OUTPUT_DIR = "data/features"

COMMAND_GET_KMERS = f"jellyfish count -m {K_SIZE} -s 10000M -t 10 {FILE_DIR}/<seq_name>/<seq_name>.fasta -o data/kmer_counter/<seq_name>.jf"
COMMAND_DUMP_DATA = "jellyfish dump -c data/kmer_counter/<seq_name>.jf > data/kmer_counter/<seq_name>.fa"

EXCLUDE_LIST = []


def main():
    file_list = os.listdir("data/raw_data")

    create_folder("data/kmer_counter")
    create_folder(OUTPUT_DIR)

    for file_name in file_list:
        sra_id = file_name
        print("RUNNING FOR SEQ: ", sra_id)

        if sra_id in EXCLUDE_LIST:
            continue
        # 538  094
        # AAAAA - 962746
        output_dict = {}

        start_time = time.time()

        os.system(COMMAND_GET_KMERS.replace("<seq_name>", sra_id))
        os.system(COMMAND_DUMP_DATA.replace("<seq_name>", sra_id))

        with open(f"data/kmer_counter/{sra_id}.fa", "r") as file:
            number = None

            for index, line in enumerate(file):
                key, value = line.strip().split(" ")
                output_dict[key] = value

        save_data_as_csv(sra_id=sra_id, data=output_dict)

        print("--- %s seconds ---" % (time.time() - start_time))


def save_data_as_csv(sra_id: str, data: dict):
    total_kmer_list = list(data.values())
    write_csv_file(f"{OUTPUT_DIR}/{sra_id}.csv", [total_kmer_list])


if __name__ == "__main__":
    main()
