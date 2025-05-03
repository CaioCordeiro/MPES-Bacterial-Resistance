import fcntl
import hashlib
import os
import os.path
import threading
import time
from typing import Dict, List, Optional

import psutil

from utils.logging_config import get_logger

from .utils import create_folder  # Assuming these are in a 'utils.py' file
from .utils import write_csv_file


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
        cache_dir: str = "data/cache",
        root_dir: str = '.',
        bac_name: str = "kleb",
        max_memory_percent: float = 70.0,
    ):
        """
        Initializes the KmerExtractor with parameters for k-mer extraction.

        Args:
            k_size: The size of the k-mers to count.
            file_dir: The directory containing the input FASTQ files.
            output_dir: The directory where the k-mer count CSV files will be saved.
            kmer_counter_dir: The directory used by Jellyfish for intermediate files.
            exclude_list: A list of SRA IDs to exclude from processing.
            max_memory_percent: Maximum percentage of system memory to use
        """
        self.root_dir = root_dir
        self.bac_name = bac_name
        self.k_size = k_size
        self.file_dir = os.path.join(self.root_dir, file_dir, self.bac_name)
        self.output_dir = os.path.join(self.root_dir, output_dir, self.bac_name)
        self.kmer_counter_dir = os.path.join(self.root_dir, kmer_counter_dir, self.bac_name)
        self.cache_dir = os.path.join(self.root_dir, cache_dir)
        self.lock_dir = os.path.join(self.cache_dir, "locks")
        self.max_memory_percent = max_memory_percent
        self.exclude_list = exclude_list
        self.logger = get_logger()
        # Calculate memory limit based on system memory
        mem_limit = int(
            (psutil.virtual_memory().total * (self.max_memory_percent / 100))
            / (1024 * 1024)
        )
        self._command_get_kmers = f"jellyfish count -m {self.k_size} -s {mem_limit}M -t 4 {self.file_dir}/<seq_name>/<seq_name>.fasta -o {self.kmer_counter_dir}/<seq_name>.jf"
        self._command_dump_data = f"jellyfish dump -c {self.kmer_counter_dir}/<seq_name>.jf > {self.kmer_counter_dir}/<seq_name>.fa"
        create_folder(os.path.join(self.root_dir, self.kmer_counter_dir))
        create_folder(os.path.join(self.root_dir, self.output_dir))
        create_folder(os.path.join(self.root_dir, self.cache_dir))
        create_folder(os.path.join(self.root_dir, self.lock_dir))

    def _acquire_lock(self, seq_name: str) -> Optional[int]:
        """
        Acquires a lock for the given sequence to prevent parallel processing.

        Args:
            seq_name: The sequence identifier to lock.

        Returns:
            Optional file descriptor if lock was acquired, None otherwise.
        """
        lock_file = os.path.join(self.root_dir, self.lock_dir, f"{seq_name}.lock")
        try:
            fd = open(lock_file, "w")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (IOError, BlockingIOError):
            # Another process is already processing this sequence
            return None

    def _release_lock(self, fd: Optional[int]) -> None:
        """Releases a previously acquired lock."""
        if fd:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def _get_cache_key(self, seq_name: str) -> str:
        """Generate a unique cache key based on the sequence name and k-mer size."""
        return hashlib.md5(f"{seq_name}_{self.k_size}".encode()).hexdigest()

    def _check_cache(self, seq_name: str) -> Dict[str, str]:
        """Check if k-mer data for the given sequence is already cached."""
        cache_key = self._get_cache_key(seq_name)
        cache_file = os.path.join(self.output_dir, f"{seq_name}.csv")
        if os.path.exists(cache_file):
            self.logger.debug(f"Using cached k-mer data for: {seq_name}")
            with open(cache_file, "r") as file:
                return {line.split(",")[0]: line.split(",")[1].strip() for line in file}
        return None

    def _save_to_cache(self, seq_name: str, data: Dict[str, str]) -> None:
        """Save k-mer data to cache."""
        cache_key = self._get_cache_key(seq_name)
        cache_file = os.path.join(self.output_dir, f"{seq_name}.csv")
        with open(cache_file, "w") as file:
            for key, value in data.items():
                file.write(f"{key},{value}\n")

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
        filepath = os.path.join(self.root_dir, self.kmer_counter_dir, f"{seq_name}.fa")
        try:
            with open(filepath, "r") as file:
                number = None
                for index, line in enumerate(file):
                    key, value = line.strip().split(" ")
                    output_dict[key] = value
        except FileNotFoundError:
            self.logger.warning(
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
        output_file = os.path.join(self.root_dir, self.output_dir, f"{sra_id}.csv")
        write_csv_file(output_file, [list(data.values())])

    def _process_batch(self, file_list: List[str]) -> None:
        for file_name in file_list:
            sra_id = file_name
            self.logger.info(f"Processing sequence: {sra_id}")

            if sra_id in self.exclude_list:
                continue

            # Check if output file already exists
            output_file = os.path.join(self.root_dir, self.output_dir, f"{sra_id}.csv")
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                self.logger.debug(f"Output file already exists for {sra_id}, skipping.")
                continue

            # cached_data = self._check_cache(sra_id)
            # if cached_data:
            #     # self._save_kmer_data(sra_id=sra_id, data=cached_data)
            #     continue

            # Try to acquire lock for this sequence
            lock_fd = self._acquire_lock(sra_id)
            if not lock_fd:
                # Another process is already processing this sequence
                self.logger.info(
                    f"Another process is already processing {sra_id}. Waiting..."
                )
                # Wait for the cache file to appear
                max_wait_time = 300  # 5 minutes timeout
                wait_time = 0
                # cache_key = self._get_cache_key(sra_id)
                # cache_file = os.path.join(self.root_dir, self.cache_dir, f"{cache_key}.csv")
                while wait_time < max_wait_time:
                    if os.path.exists(output_file):
                        self.logger.info(
                            f"Sequence {sra_id} is now processed by another thread."
                        )
                        cached_data = self._check_cache(sra_id)
                        if cached_data:
                        #     self._save_kmer_data(sra_id=sra_id, data=cached_data)
                            return
                    time.sleep(5)
                    wait_time += 5
                self.logger.warning(
                    f"Timeout waiting for {sra_id} to be processed by another thread."
                )
                continue

            try:
                start_time = time.time()

                # Check if we have enough memory before processing
                available_memory_percent = psutil.virtual_memory().percent
                if available_memory_percent < 20:  # Less than 20% memory available
                    self.logger.warning(
                        f"Low memory ({available_memory_percent:.1f}% available). Waiting before processing {sra_id}..."
                    )
                    time.sleep(60)  # Wait for a minute to let other processes finish

                self._run_jellyfish_commands(sra_id)
                kmer_data = self._parse_jellyfish_output(sra_id)
                # self._save_to_cache(sra_id, kmer_data)
                self._save_kmer_data(sra_id=sra_id, data=kmer_data)
                self.logger.debug(
                    f"Processing {sra_id} took {time.time() - start_time:.2f} seconds"
                )
            finally:
                self._release_lock(lock_fd)

    def process_sequences(self) -> None:
        """
        Iterates through the files in the specified input directory, extracts k-mers
        using Jellyfish, and saves the counts to CSV files.
        """
        file_list = os.listdir(self.file_dir)

        # Process files in batches to control memory usage
        batch_size = 5  # Process 5 files at a time
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i : i + batch_size]
            self._process_batch(batch)
            # Force garbage collection after each batch
            import gc

            gc.collect()
