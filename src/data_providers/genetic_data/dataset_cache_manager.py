import fcntl
import hashlib
import json
import os
import time
from typing import Dict, Optional

import pandas as pd

from utils.logging_config import get_logger


class DatasetCacheManager:
    """
    Manages caching of datasets to avoid redundant processing.
    Provides thread-safe access to cached datasets.
    """

    def __init__(self, cache_dir: str = "data/cache", bac_name: str = "kleb"):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached datasets
            bac_name: Name of the bacteria being processed
        """
        self.cache_dir = cache_dir
        self.bac_name = bac_name
        self.lock_dir = os.path.join(cache_dir, "locks")

        self.logger = get_logger()

        # Create directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)

        # Cache file paths
        self.dataset_cache_file = os.path.join(
            self.cache_dir, f"{self._get_cache_key()}.csv"
        )
        self.metadata_file = os.path.join(
            self.cache_dir, f"{self._get_cache_key()}.meta"
        )
        self.lock_file = os.path.join(self.lock_dir, f"{self.bac_name}_dataset.lock")

    def _get_cache_key(self) -> str:
        """Generate a unique cache key for the dataset."""
        return hashlib.md5(f"{self.bac_name}_dataset".encode()).hexdigest()

    def _acquire_lock(self) -> Optional[int]:
        """
        Acquire a lock for thread-safe cache access.

        Returns:
            File descriptor if lock acquired, None otherwise
        """
        try:
            fd = open(self.lock_file, "w")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (IOError, BlockingIOError):
            return None

    def _release_lock(self, fd: Optional[int]) -> None:
        """Release a previously acquired lock."""
        if fd:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata about the cached dataset."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)

    def _load_metadata(self) -> Optional[Dict]:
        """Load metadata about the cached dataset."""
        if not os.path.exists(self.metadata_file):
            return None

        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except:
            return None

    def dataset_exists(self) -> bool:
        """
        Check if a valid cached dataset exists.

        Returns:
            True if a valid cache exists, False otherwise
        """
        if not os.path.exists(self.dataset_cache_file):
            return False

        if os.path.getsize(self.dataset_cache_file) <= 1:
            return False

        # Check metadata for cache validity
        metadata = self._load_metadata()
        if metadata is None:
            return False

        # Add any additional validation logic here
        # For example, check if cache is too old
        cache_time = metadata.get("timestamp", 0)
        cache_age = time.time() - cache_time
        max_age = 7 * 24 * 60 * 60  # 7 days in seconds

        if cache_age > max_age:
            self.logger.debug(
                f"Cache for {self.bac_name} is {cache_age/86400:.1f} days old, exceeding {max_age/86400:.1f} days limit"
            )
            return False

        return True

    def save_dataset(self, df: pd.DataFrame) -> bool:
        """
        Save a dataset to cache.

        Args:
            df: DataFrame to cache

        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty:
            return False

        lock_fd = self._acquire_lock()
        if not lock_fd:
            self.logger.warning(
                f"Could not acquire lock to save dataset cache for {self.bac_name}"
            )
            return False

        try:
            # Save the dataset with optimized settings
            df.to_csv(self.dataset_cache_file, index=False, compression="infer")
            self.logger.info(
                f"Saved dataset to cache: {self.dataset_cache_file} ({len(df)} rows)"
            )

            # Save metadata
            metadata = {
                "timestamp": time.time(),
                "rows": len(df),
                "columns": len(df.columns),
                "bacteria": self.bac_name,
            }
            self._save_metadata(metadata)
            self.logger.info(
                f"Saved dataset to cache: {self.dataset_cache_file} ({len(df)} rows)"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error saving dataset to cache: {e}")
            return False
        finally:
            self._release_lock(lock_fd)

    def load_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load a dataset from cache.

        Returns:
            DataFrame if cache exists and is valid, None otherwise
        """
        if not self.dataset_exists():
            return None

        try:
            # Use optimized pandas read_csv settings
            df = pd.read_csv(
                self.dataset_cache_file,
                low_memory=False,
                compression="infer",
                memory_map=True,
            )

            if df.empty:
                self.logger.warning(f"Cached dataset is empty for {self.bac_name}")
                return None

            self.logger.info(
                f"Loaded dataset from cache: {self.dataset_cache_file} ({len(df)} rows)"
            )
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset from cache: {e}")
            return None
