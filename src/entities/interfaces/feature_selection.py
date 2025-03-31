from abc import ABC, abstractmethod

from pandas import DataFrame


class FeatureSelectionInterface(ABC):
    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def fit(self, data: DataFrame) -> DataFrame:
        pass
