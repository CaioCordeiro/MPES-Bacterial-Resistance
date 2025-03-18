import os
from typing import List


class SraDownloader:
    """
    A utility class to download FASTQ files from the NCBI SRA database
    using the `fasterq-dump` tool.
    """

    def __init__(
        self,
        sra_id_list: List[str],
        exclude_list: List[str] = [],
        output_dir: str = "data/raw_data/",
    ):
        """
        Initializes the SraDownloader with a list of SRA IDs and optional
        exclusion list and output directories.

        Args:
            sra_id_list: A list of SRA experiment identifiers to download.
            exclude_list: A list of SRA experiment identifiers to exclude from download.
            output_dir: The directory where downloaded FASTQ files will be saved.
        """
        self.sra_id_list = list(set(sra_id_list) - set(exclude_list))
        self.exclude_list = exclude_list
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _download_file(self, experiment_id: str) -> None:
        """
        Downloads a FASTQ file for a given SRA experiment ID using `fasterq-dump`.

        Args:
            experiment_id: The SRA experiment identifier to download.
        """
        try:
            command = f"fasterq-dump {experiment_id} -O {self.output_dir}/{experiment_id} --fasta -p -t temp/"
            os.system(command)
        except Exception as error:
            print(f"Error downloading file for: {experiment_id}\n{error}")

    def download(self) -> None:
        """
        Iterates through the list of SRA IDs and downloads the corresponding
        FASTQ files.
        """
        download_counter = 0

        for experiment_id in self.sra_id_list:
            download_counter += 1
            print(
                f"Downloading file: {download_counter}/{len(self.sra_id_list)}",
                end="\t",
            )
            self._download_file(experiment_id)
            print(
                f"\nFile {download_counter}/{len(self.sra_id_list)} download finished"
            )
