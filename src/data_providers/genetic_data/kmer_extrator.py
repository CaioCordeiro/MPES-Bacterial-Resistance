import os
import time
from typing import Dict, List

from .utils import (create_folder,  # Assuming these are in a 'utils.py' file
                    write_csv_file)


class KmerExtractor:
    """
    A class to extract k-mers from FASTQ files located in a specified directory
    using Jellyfish.
    """

    def __init__(
        self,
        k_size: int = 5,
        file_dir: str = "data/raw_data",
        output_dir: str = "data/features",
        kmer_counter_dir: str = "data/kmer_counter",
        exclude_list: List[str] = [],
    ):
        """
        Initializes the KmerExtractor with parameters for k-mer extraction.

        Args:
            k_size: The size of the k-mers to count.
            file_dir: The directory containing the input FASTQ files.
            output_dir: The directory where the k-mer count CSV files will be saved.
            kmer_counter_dir: The directory used by Jellyfish for intermediate files.
            exclude_list: A list of SRA IDs to exclude from processing.
        """
        self.k_size = k_size
        self.file_dir = file_dir
        self.output_dir = output_dir
        self.kmer_counter_dir = kmer_counter_dir
        self.exclude_list = exclude_list
        self._command_get_kmers = f"jellyfish count -m {self.k_size} -s 10000M -t 10 {self.file_dir}/<seq_name>/<seq_name>.fasta -o {self.kmer_counter_dir}/<seq_name>.jf"
        self._command_dump_data = f"jellyfish dump -c {self.kmer_counter_dir}/<seq_name>.jf > {self.kmer_counter_dir}/<seq_name>.fa"
        create_folder(self.kmer_counter_dir)
        create_folder(self.output_dir)

    def _run_jellyfish_commands(self, seq_name: str) -> None:
        """
        Runs the Jellyfish count and dump commands for a given sequence name.

        Args:
            seq_name: The identifier of the sequence (e.g., SRA ID).
        """
        get_kmers_command = self._command_get_kmers.replace("<seq_name>", seq_name)
        dump_data_command = self._command_dump_data.replace("<seq_name>", seq_name)
        os.system(get_kmers_command)
        os.system(dump_data_command)

    def _parse_jellyfish_output(self, seq_name: str) -> Dict[str, str]:
        """
        Parses the Jellyfish dump output file to create a dictionary of k-mers and their counts.

        Args:
            seq_name: The identifier of the sequence.

        Returns:
            A dictionary where keys are k-mers and values are their counts.
        """
        output_dict: Dict[str, str] = {}
        filepath = f"{self.kmer_counter_dir}/{seq_name}.fa"
        try:
            with open(filepath, "r") as file:
                number = None
                for index, line in enumerate(file):
                    key, value = line.strip().split(" ")
                    output_dict[key] = value
        except FileNotFoundError:
            print(
                f"Warning: Jellyfish output file not found for {seq_name}: {filepath}"
            )
        return output_dict

    def _save_kmer_data(self, sra_id: str, data: Dict[str, str]) -> None:
        """
        Saves the extracted k-mer counts to a CSV file.

        Args:
            sra_id: The SRA identifier of the sequence.
            data: A dictionary containing k-mers and their counts.
        """
        total_kmer_list = list(data.values())
        write_csv_file(f"{self.output_dir}/{sra_id}.csv", [total_kmer_list])

    def process_sequences(self) -> None:
        """
        Iterates through the files in the specified input directory, extracts k-mers
        using Jellyfish, and saves the counts to CSV files.
        """
        file_list = os.listdir(self.file_dir)

        for file_name in file_list:
            sra_id = file_name
            print("RUNNING FOR SEQ: ", sra_id)

            if sra_id in self.exclude_list:
                continue

            start_time = time.time()

            self._run_jellyfish_commands(sra_id)
            kmer_data = self._parse_jellyfish_output(sra_id)
            print(kmer_data)
            self._save_kmer_data(sra_id=sra_id, data=kmer_data)

            print(f"--- {time.time() - start_time} seconds ---")
