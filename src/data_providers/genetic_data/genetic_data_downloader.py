import os
from typing import Generator, List


class SraDownloader:
    """
    A utility class to download FASTQ files from the NCBI SRA database
    using the `fasterq-dump` tool, implemented as a generator.
    """

    def __init__(
        self,
        sra_id_list: List[str],
        exclude_list: List[str] = [],
        output_dir: str = "data/raw_data/",
        bac_name: str = "kleb",
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
        self.bac_name = bac_name
        os.makedirs(self.output_dir, exist_ok=True)

    def _download_file(self, experiment_id: str) -> None:
        """
        Downloads a FASTQ file for a given SRA experiment ID using `fasterq-dump`.

        Args:
            experiment_id: The SRA experiment identifier to download.
        """
        try:
            output_path = os.path.join(self.output_dir, self.bac_name, experiment_id)
            print(f"Downloading on: {output_path}")
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            command = f"fasterq-dump {experiment_id} -O {output_path} --fasta -p -t {temp_dir}/"
            os.system(command)
        except Exception as error:
            print(f"Error downloading file for: {experiment_id}\n{error}")

    def download_generator(self) -> Generator[str, None, None]:
        """
        Iterates through the list of SRA IDs and downloads the corresponding
        FASTQ files one by one, yielding the downloaded experiment ID.

        Yields:
            str: The SRA experiment ID that was just downloaded.
        """
        total_files = len(self.sra_id_list)
        for i, experiment_id in enumerate(self.sra_id_list, 1):
            print(f"Downloading file: {i}/{total_files}\t", end="")
            self._download_file(experiment_id)
            print(f"\nFile {i}/{total_files} download finished")
            yield experiment_id

    def download(self) -> None:
        """
        Downloads all FASTQ files by iterating through the generator.
        This method is provided for backward compatibility or convenience.
        """
        for _ in self.download_generator():
            pass
