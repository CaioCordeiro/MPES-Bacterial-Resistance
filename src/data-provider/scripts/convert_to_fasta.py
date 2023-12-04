# Config
from config import Config

#  Utils
import os

COMMAND = "fastq_to_fasta -i data/raw_data/<seq>/<seq>_1.fastq -o  data/fasta_files/<seq>.fasta"


def main():
    raw_folders = os.listdir('data/raw_data')

    for folder_name in raw_folders:
        print("RUNNING FOR SEQ: ", folder_name)

        if folder_name in Config.EXCLUDE_LIST:
            continue

        os.system(COMMAND.replace("<seq>", folder_name))


if __name__ == "__main__":
    main()
