# Utils
import sys, os

# url = "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={}&format=fasta".format(
#     experiment_id)

# Configuration file

SRA_ID_LIST = ['SRR5386221', 'SRR5386094']

EXCLUDE_LIST = []

list(set(SRA_ID_LIST) - set(EXCLUDE_LIST))



def download_file(experiment_id):
    try:
        os.system(
            "fasterq-dump {} -O data/raw_data/{} --fasta -p -t temp/".format(experiment_id, experiment_id))
    except Exception as error:
        print('Error downloading file for: ', experiment_id, "\n", error)


def main():
    experiment_counter = 0
    download_counter = 0

    for experiment_id in SRA_ID_LIST:
        download_counter += 1

        print(f"Downloading file: {download_counter}/{len(SRA_ID_LIST)}".format(end="\t"))
        download_file(experiment_id)
        print(f"\nFile {download_counter}/{len(SRA_ID_LIST)} download finished")


if __name__ == "__main__":
    main()
