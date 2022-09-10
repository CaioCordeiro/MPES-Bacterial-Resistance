from abc import ABC, abstractmethod

from pandas import DataFrame
from sklean import Model


class ModelInterface(ABC):
    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def algorithmImplementation(self, data: DataFrame) -> Model:
        pass
