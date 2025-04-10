from sklearn.model_selection import cross_val_score

import constants.constants as const
from data_providers.genetic_data.genetic_data import GeneticDataset
from entities.interfaces.feature_selection import FeatureSelectionInterface


class GeneticDataRun:
    def __init__(
        self,
        data: GeneticDataset,
        target: str,
        bac: str,
        model,
        fs: FeatureSelectionInterface,
    ):
        self.data = data
        self.target = target
        self.bac = bac
        self.model = model
        self.fs = fs

    @property
    def _df(self):
        anti_list_without_target = const.ANTIBIOTIC_LIST.copy()
        print(const.ANTIBIOTIC_LIST)
        anti_list_without_target.remove(self.target)
        return self.data.treated_data.drop(
            anti_list_without_target, axis=1, errors="ignore"
        )

    @property
    def _X(self):
        return self._df.drop(self.target, axis=1, errors="ignore")

    @property
    def _y(self):
        return self._df[self.target]

    @property
    def _selected_features(self):
        print(f"Starting Feature Selection: {self.fs.name} on bac {self.bac}")
        return self.fs.fit(self._df, self._X, self._y)

    def run(self):
        df = self._selected_features
        scores = cross_val_score(self.model, df, self._y, cv=10)
        return scores
