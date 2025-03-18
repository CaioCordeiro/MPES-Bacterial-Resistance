from abc import ABC, abstractmethod
from typing import Any

from pandas import DataFrame


class DatasetInterface(ABC):
    """
    An abstract base class defining the interface for dataset objects.

    Subclasses should implement methods for accessing treated data,
    splitting the dataset, and providing metrics.
    """

    def __init__(self, raw_data: Any, name: str, metric_provider: Any = None):
        """
        Initializes the DatasetInterface.

        Args:
            raw_data: The raw, unprocessed data. The type can vary depending on the dataset.
            name: A descriptive name for the dataset.
            metric_provider: An optional object that can provide metrics related to the dataset.
        """
        self.raw_data = raw_data
        self.name = name
        self.metric_provider = metric_provider

    @property
    @abstractmethod
    def treated_data(self) -> DataFrame:
        """
        Abstract property that returns the treated (processed, cleaned, feature-engineered)
        data as a Pandas DataFrame.

        Returns:
            DataFrame: The treated dataset.
        """
        pass

    @abstractmethod
    def splitted_dataset(self, test_size: float) -> list[DataFrame]:
        """
        Abstract method that splits the treated dataset into training and testing sets.

        Args:
            test_size: The proportion of the dataset to include in the test split (0.0 to 1.0).

        Returns:
            list[DataFrame]: A list containing two DataFrames: the training set and the testing set.
        """
        pass

    @property
    @abstractmethod
    def metrics(self):
        """
        Abstract property that provides access to the metrics provider object.

        Returns:
            Any: The metrics provider object, or None if not set.
        """
        pass
