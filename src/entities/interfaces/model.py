from typing import Optional
import numpy as np

class ModelInterface:
    """
    Interface for machine learning models.
    """

    def fit(self, X, y):
        """
        Fits the model to the provided data.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Makes predictions using the fitted model.
        """
        raise NotImplementedError

    def get_model_summary(self) -> dict:
        """
        Returns a summary of the model's performance and configuration.
        """
        raise NotImplementedError

    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """
        Returns the confusion matrix of the model, if applicable.
        """
        pass
