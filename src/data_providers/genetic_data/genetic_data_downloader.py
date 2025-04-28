import fcntl
import os
import random
import time
from typing import Dict, Generator, List, Optional

import psutil

from utils.logging_config import get_logger


class SraDownloader:
    """
    A utility class to download FASTQ files from the NCBI SRA database
    using the `fasterq-dump` tool, implemented as a generator.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        sra_id_list: List[str] = [],
        exclude_list: List[str] = [],
        output_dir: str = "data/raw_data/",
        max_sra_ids: Optional[int] = None,
        bac_name: str = "kleb",
        max_concurrent_downloads: int = 3,
    ):
        """
        Initializes the SraDownloader with a list of SRA IDs and optional
        exclusion list and output directories.

        Args:
            sra_id_list: A list of SRA experiment identifiers to download.
            exclude_list: A list of SRA experiment identifiers to exclude from download.
            output_dir: The directory where downloaded FASTQ files will be saved.
            max_sra_ids: Maximum number of SRA IDs to process. If None, all IDs are processed.
            bac_name: Name of the bacteria being processed.
            max_concurrent_downloads: Maximum number of concurrent downloads
        """
        # Filter out excluded IDs
        filtered_ids = list(set(sra_id_list) - set(exclude_list))
        # Limit the number of SRA IDs if max_sra_ids is specified
        if max_sra_ids is not None and max_sra_ids < len(filtered_ids):
            filtered_ids = random.sample(filtered_ids, max_sra_ids)
        self.sra_id_list = filtered_ids
        self.exclude_list = exclude_list
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.lock_dir = os.path.join(self.cache_dir, "locks")
        self.bac_name = bac_name
        self.max_concurrent_downloads = max_concurrent_downloads
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)
        self.logger = get_logger()

    def _acquire_lock(self, experiment_id: str) -> Optional[int]:
        """
        Acquires a lock for the given experiment ID to prevent parallel downloads.

        Args:
            experiment_id: The SRA experiment identifier to lock.

        Returns:
            Optional file descriptor if lock was acquired, None otherwise.
        """
        lock_file = os.path.join(self.lock_dir, f"{experiment_id}.lock")
        try:
            fd = open(lock_file, "w")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (IOError, BlockingIOError):
            # Another process is already downloading this file
            return None

    def _release_lock(self, fd: int) -> None:
        """
        Releases a previously acquired lock.

        Args:
            fd: File descriptor of the lock file.
        """
        if fd:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def _check_active_downloads(self) -> int:
        """Count the number of active fasterq-dump processes."""
        import subprocess

        result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
        output = result.stdout
        active_downloads = output.count("fasterq-dump")
        return active_downloads

    def _download_file(self, experiment_id: str) -> None:
        """
        Downloads a FASTQ file for a given SRA experiment ID using `fasterq-dump`.

        Args:
            experiment_id: The SRA experiment identifier to download.
        """
        cache_file = os.path.join(self.cache_dir, f"{experiment_id}.fasta")
        if os.path.exists(cache_file):
            self.logger.debug(f"Using cached file for: {experiment_id}")
            return

        # Try to acquire lock for this experiment
        lock_fd = self._acquire_lock(experiment_id)
        if not lock_fd:
            # Another process is already downloading this file
            self.logger.info(
                f"Another process is already downloading {experiment_id}. Waiting..."
            )
            # Wait for the file to appear in cache
            max_wait_time = 600  # 10 minutes timeout
            wait_time = 0
            while wait_time < max_wait_time:
                if os.path.exists(cache_file):
                    self.logger.info(f"File {experiment_id} is now available in cache.")
                    return
                time.sleep(10)
                wait_time += 10
            self.logger.warning(
                f"Timeout waiting for {experiment_id} to be downloaded by another process."
            )
            return

        try:
            # Check if we have too many concurrent downloads
            while self._check_active_downloads() >= self.max_concurrent_downloads:
                self.logger.info(
                    f"Too many concurrent downloads ({self._check_active_downloads()}). Waiting..."
                )
                time.sleep(30)

            # Check memory usage before starting download
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:  # More than 80% memory used
                self.logger.warning(
                    f"High memory usage ({memory_percent}%). Waiting before downloading {experiment_id}..."
                )
                time.sleep(60)  # Wait for a minute to let memory free up

            # Check disk space
            disk_usage = psutil.disk_usage(
                os.path.dirname(os.path.abspath(self.output_dir))
            )
            if disk_usage.percent > 90:  # More than 90% disk used
                self.logger.warning(
                    f"Low disk space ({disk_usage.percent}%). This may cause download failures."
                )

            output_path = os.path.join(self.output_dir, self.bac_name, experiment_id)
            self.logger.info(f"Downloading on: {output_path}")
            temp_dir = "temp"
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)
            # os.system(f"prefetch {experiment_id} -O {temp_dir}/pre_fetch/")
            # Use smaller hash table size and limit threads to reduce memory usage
            command = f"fasterq-dump {experiment_id} -O {output_path} --fasta -p -t {temp_dir}/"
            os.system(command)

            # Cache the downloaded filem,n
            os.system(f"cp {output_path}/{experiment_id}.fasta {cache_file}")
        except Exception as error:
            self.logger.error(f"Error downloading file for: {experiment_id}\n{error}")
        finally:
            self._release_lock(lock_fd)

    def download_generator(self) -> Generator[str, None, None]:
        """
        Iterates through the list of SRA IDs and downloads the corresponding
        FASTQ files one by one, yielding the downloaded experiment ID.

        Yields:
            str: The SRA experiment ID that was just downloaded.
        """
        total_files = len(self.sra_id_list)
        for i, experiment_id in enumerate(self.sra_id_list, 1):
            self.logger.info(f"Downloading file: {i}/{total_files}")
            self._download_file(experiment_id)
            self.logger.info(f"File {i}/{total_files} download finished")
            yield experiment_id

    def download(self) -> None:
        """
        Downloads all FASTQ files by iterating through the generator.
        This method is provided for backward compatibility or convenience.
        """
        for _ in self.download_generator():
            pass
