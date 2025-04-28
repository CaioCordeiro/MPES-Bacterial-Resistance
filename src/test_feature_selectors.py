import numpy as np
import pandas as pd

from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from feature_selection.pearson_correlation import PearsonCorrelationSelector
from feature_selection.rrelief import RReliefF
from utils.logging_config import get_logger, setup_logging


def create_test_data(n_samples=100, n_features=20):
    """Create synthetic data for testing feature selectors"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Make some features more important
    y = 5 * X[:, 0] + 3 * X[:, 1] + X[:, 2] + np.random.randn(n_samples)

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def test_feature_selector(selector_class, n_features=5):
    """Test a feature selector with synthetic data"""
    logger = get_logger()
    logger.info(f"Testing {selector_class.__name__} with {n_features} features")

    # Create test data
    df = create_test_data()

    # Initialize selector
    selector = selector_class(n_features_to_select=n_features)

    try:
        # Fit selector
        result = selector.fit(df, "target")

        # Check if we got the right number of features
        selected_features = [col for col in result.columns if col != "target"]
        logger.info(f"Selected features: {selected_features}")

        if len(selected_features) != n_features:
            logger.error(
                f"Expected {n_features} features, got {len(selected_features)}"
            )
        else:
            logger.info(
                f"SUCCESS: {selector_class.__name__} selected {n_features} features as expected"
            )

        # Check if feature_scores_ attribute exists and has values
        if (
            hasattr(selector, "feature_scores_")
            and selector.feature_scores_ is not None
        ):
            logger.info(
                f"Feature scores available: {len(selector.feature_scores_)} features scored"
            )
            top_features = selector.feature_scores_.sort_values(ascending=False).head(5)
            logger.info(f"Top 5 features by score: {top_features}")
        else:
            logger.error(
                f"No feature_scores_ attribute found in {selector_class.__name__}"
            )

        return True
    except Exception as e:
        logger.error(f"Error testing {selector_class.__name__}: {str(e)}")
        return False


def main():
    # Setup logging
    setup_logging("INFO")
    logger = get_logger()
    logger.info("Starting feature selector tests")

    # Test each feature selector
    selectors = [RReliefF, PearsonCorrelationSelector, BanzhafFeatureSelector]

    all_passed = True
    for selector_class in selectors:
        passed = test_feature_selector(selector_class)
        all_passed = all_passed and passed

    if all_passed:
        logger.info("All feature selectors passed basic tests")
    else:
        logger.error("Some feature selectors failed tests")


if __name__ == "__main__":
    main()
