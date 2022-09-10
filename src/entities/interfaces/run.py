from abc import ABC

from src.entities.interfaces.dataset import DatasetInterface
from src.entities.interfaces.feature_selection import FeatureSelectionInterface
from src.entities.interfaces.metrics_calculator import \
    MetricsCalculatorInterface
from src.entities.interfaces.model import ModelInterface


class Run(ABC):
    def __init__(self, name: str, metric_provider: MetricsCalculatorInterface, model: ModelInterface,
                 feature_selector: FeatureSelectionInterface, dataset: DatasetInterface):
        self.name = name
        self.metric_provider = metric_provider
        self.model = model
        self.feature_selector = feature_selector
        self.dataset = dataset
