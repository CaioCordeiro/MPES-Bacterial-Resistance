from abc import ABC, abstractmethod, abstractproperty

from pandas import DataFrame


class DatasetInterface(ABC):
    def __init__(self, raw_data, name, metric_provider):
        self.raw_data = raw_data
        self.name = name
        self.metric_provider = metric_provider

    @abstractproperty
    def treated_data(self) -> DataFrame:
        pass

    @abstractmethod
    def splitted_dataset(self, test_size) -> list[DataFrame]:
        pass

    @abstractproperty
    def metrics(self):
        pass
