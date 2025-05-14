import pandas as pd
import numpy as np
import shap
import constants.constants as const
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from typing import List, Optional
from entities.interfaces.feature_selection import FeatureSelectionInterface


class ShapFeatureSelector(FeatureSelectionInterface):
    """
    Performs feature ranking based on SHAP values.
    """

    def __init__(
        self, model, train_model_before_shap: bool = True, n_features_to_select: int = 0
    ):
        """
        Initializes the ShapFeatureSelector.

        Args:
            model: A machine learning model compatible with SHAP.
            train_model_before_shap: Whether to train the model before SHAP evaluation.
        """
        super().__init__(name="SHAPFeatureSelector")
        self.model = model
        self.train_model_before_shap = train_model_before_shap
        self.feature_scores_: Optional[dict] = None
        self.ranked_features_: Optional[List[str]] = None
        self.n_features_to_select = n_features_to_select

    def fit(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """
        Ranks features based on SHAP values.

        Args:
            df: Input DataFrame containing features and the target variable.
            target_column: Name of the target column in the DataFrame.

        Returns:
            A ranked list of feature names.
        """
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in the DataFrame."
            )

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # Scale X data
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        print("X shape:", X.shape)
        print("X columns:", X.columns)
        print("X head:\n", X.head())
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Train the model if required
        if self.train_model_before_shap:
            self.model.fit(X_train, y_train)
            fitted_model = self.model
        else:
            try:
                check_is_fitted(self.model)
            except Exception as e:
                raise ValueError(
                    "The provided model is not fitted. Set train_model_before_shap=True to train it within the selector."
                ) from e
            fitted_model = self.model
        print("Target value counts:", y.value_counts())
        print("Feature variance:\n", X.var())
        print("Model predictions on X:", self.model.predict(X_test))
        # Model accuracy on the test set
        from sklearn.metrics import accuracy_score

        y_pred = fitted_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on test set: {accuracy}")

        def model_predict(data):
            # Ensure the input data has the same feature names as the training data
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=X.columns)
            return fitted_model.predict(data)

        sampled_X = shap.kmeans(X, 30).data
        # Calculate SHAP values
        explainer = shap.KernelExplainer(model_predict, sampled_X)
        shap_values = explainer(sampled_X)

        # Compute mean absolute SHAP values for each feature
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        self.feature_scores_ = dict(zip(X.columns, feature_importance))

        # Make sure that the most important features are ranked first
        self.ranked_features_ = sorted(
            X.columns, key=self.feature_scores_.get, reverse=True
        )
        # Print {rank} - {feature} - {score}
        for rank, feature in enumerate(self.ranked_features_, start=1):
            print(f"{rank} - {feature} - {self.feature_scores_[feature]}")
        # save the feature : feature importance score into a file
        feature_importance_df = pd.DataFrame(
            list(self.feature_scores_.items()), columns=["Feature", "Importance"]
        )
        # timestamp the file name
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        feature_importance_df.to_csv(f"{timestamp}_feature_importance.csv", index=False)
        return self.ranked_features_

    def get_feature_scores(self) -> dict:
        """
        Returns the feature importance scores.

        Returns:
            A dictionary of feature importance scores.
        """
        if self.feature_scores_ is None:
            raise ValueError(
                "Feature scores have not been calculated. Call fit() first."
            )
        return self.feature_scores_
