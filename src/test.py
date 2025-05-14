from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split

from constants.constants import ROOT_DIR
from data_providers.genetic_data.genetic_data import GeneticDataset
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from feature_selection.shap_feature_selector import ShapFeatureSelector
from models.linear_regression_model import LinearRegressionModel
from models.svm import SupportVectorMachineModel
from runs.genetic_data_run import GeneticDataRun

anti = [
    "amikacin",
    "ampicillin",
    "ampicillin/sulbactam",
    "aztreonam",
    "cefazolin",
    "cefepime",
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "cefuroxime sodium",
    "gentamicin",
    "imipenem",
    "levofloxacin",
    "meropenem",
    "nitrofurantoin",
    "piperacillin/tazobactam",
    "tetracycline",
    "tobramycin",
    "trimethoprim/sulfamethoxazole",
    "vancomycin",
]

bac = "kleb"
data = GeneticDataset(bac_name=bac, max_sra_ids=150, root_dir="/mnt/d")
model = SupportVectorMachineModel()
df = data.treated_data.copy()
df = df.drop(anti, axis=1, errors="ignore")
X = df.drop(columns=["ciprofloxacin"]).dropna()
y = df["ciprofloxacin"].dropna()
scores = model.cross_validate(X, y, cv=10)
print(f"Cross-validation scores: {scores}")
model.fit(X, y)
print(model.get_summary())
selector = ShapFeatureSelector(model)
# print(y)
# from models.svm import SupportVectorMachineModel
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(model.get_summary())
# scores = model.cross_validate(X, y, cv=10)
# print(f"Cross-validation scores: {scores}")
# print(f"Mean cross-validation score: {scores.mean()}")
# # Accuracy on the test set
# y_pred = model.predict(X_test)
# accuracy = model.model.score(X_test, y_test)
# print(f"Test set accuracy: {accuracy}")
# # F1 score on the test set
# from sklearn.metrics import f1_score
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"F1 score on the test set: {f1}")

# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)
# accuracy = svm_model.score(X_test, y_test)
# print(f"SVM Test set accuracy: {accuracy}")
# # F1 score on the test set
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"SVM F1 score on the test set: {f1}")

ranked_features = selector.fit(df, "ciprofloxacin")
max_features_to_select = 1023
# Create a Process pool for each feature subset size
n_features_to_select = 10
from multiprocessing import Pool
from functools import partial


def run_genetic_data_run(n_features_to_select):
    run = GeneticDataRun(
        data,
        "ciprofloxacin",
        bac,
        model,
        fs=None,
        selected_features=ranked_features[:n_features_to_select],
    )
    return run.run()


with Pool() as pool:
    results = pool.map(run_genetic_data_run, range(10, max_features_to_select, 10))
# Print the results
for n_features, result in enumerate(results, start=1):
    print(f"Results for {n_features} features: {result}")
# save all results on a file
import pandas as pd

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
