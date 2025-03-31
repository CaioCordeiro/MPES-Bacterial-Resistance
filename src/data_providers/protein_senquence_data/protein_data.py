from typing import Any, List

import pandas as pd
from sklearn.model_selection import train_test_split

from constants.constants import FILE_DIR
from entities.interfaces.dataset import DatasetInterface

FILE_MAP = {
    "ac": {
        "aac": "Ac_Sa_Ca_Kl_Ec/aac_all.csv",
        "bla": "Ac_Sa_Ca_Kl_Ec/bla_all.csv",
        "dfr": "Ac_Sa_Ca_Kl_Ec/dfr_all.csv",
    },
    "ps": {
        "aac": "Ps_Vb_En/aac_all.csv",
        "bla": "Ps_Vb_En/bla_all.csv",
        "dfr": "Ps_Vb_En/dfr_all.csv",
    },
}


class ProtainDataset(DatasetInterface):
    """
    A class to load, preprocess, and represent protein-related datasets,
    adhering to the DatasetInterface.
    """

    def __init__(
        self,
        bac: str,
        ds_name: str,
        name: str = "ProtainDataset",
        metric_provider: Any = None,
    ):
        """
        Initializes the ProtainDataset.

        Args:
            bac: The identifier for the bacteria type ('ac' or 'ps').
            ds_name: The identifier for the dataset name ('aac', 'bla', or 'dfr').
            name: A descriptive name for the dataset.
            metric_provider: An optional object to provide metrics.
        """
        self.bac = bac
        self.ds_name = ds_name
        self.filepath_config = FILE_DIR
        raw_df = self._get_single_dataframe(bac, ds_name)
        super().__init__(raw_data=raw_df, name=name, metric_provider=metric_provider)
        self._treated_data: pd.DataFrame = self._preprocess_data(raw_df)

    def _get_single_dataframe(self, bac: str, ds_name: str) -> pd.DataFrame:
        """Loads a single raw dataframe based on bacteria and dataset name."""
        try:
            filepath = f"{self.filepath_config}/{FILE_MAP[bac][ds_name]}"
            return pd.read_csv(filepath)
        except Exception as err:
            raise ValueError(
                f"Invalid bac: {bac} or ds_name: {ds_name}. \n Original error: \n {str(err)}"
            )

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretizes the numerical columns of the dataframe."""
        for i in df.columns:
            if i != "Feature" and i != "Output":
                df[i] = pd.qcut(
                    df[i], q=5, labels=False, precision=0, duplicates="drop"
                )
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the preprocessing steps to the raw dataframe."""
        df = self._discretize(df)
        y = df["Output"].replace({-1: 0})
        X = df.drop(["Output", "Feature"], axis=1)
        return pd.concat([X, y], axis=1)

    @property
    def treated_data(self) -> pd.DataFrame:
        """
        Returns the treated protein dataset as a Pandas DataFrame.
        """
        return self._treated_data

    def splitted_dataset(
        self, test_size: float, random_state: int = None
    ) -> List[pd.DataFrame]:
        """
        Splits the treated protein dataset into training and testing sets.

        Args:
            test_size: The proportion of the dataset to use for the test set (e.g., 0.2 for 20%).
            random_state: Optional random seed for reproducibility.

        Returns:
            List[pd.DataFrame]: A list containing two DataFrames: [train_df, test_df].
        """
        if self._treated_data is None:
            raise ValueError("Treated data is not available.")
        train_df, test_df = train_test_split(
            self._treated_data, test_size=test_size, random_state=random_state
        )
        return [train_df, test_df]

    @property
    def metrics(self):
        """
        Returns the metrics provider object (if set).
        """
        return self.metric_provider
