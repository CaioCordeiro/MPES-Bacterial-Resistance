from sklearn.model_selection import train_test_split

import constants.constants as const
from data_providers.genetic_data.genetic_data import GeneticDataset
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
from models.linear_regression_model import LinearRegressionModel
from models.svm import SupportVectorMachineModel
from runs.genetic_data_run import GeneticDataRun

anti = [
    "Amikacin",
    "Ampicillin",
    "Ampicillin/Sulbactam",
    "Aztreonam",
    "Cefazolin",
    "Cefepime",
    "Cefoxitin",
    "Ceftazidime",
    "Ceftriaxone",
    "Cefuroxime sodium",
    "Ciprofloxacin",
    "Gentamicin",
    "Imipenem",
    "Levofloxacin",
    "Meropenem",
    "Nitrofurantoin",
    "Piperacillin/Tazobactam",
    "Tetracycline",
    "Tobramycin",
    "Trimethoprim/Sulfamethoxazole",
    "Vancomycin",
]

bac = "kleb"
data = GeneticDataset(bac_name=bac, max_sra_ids=30)
selector = BanzhafFeatureSelector(5, 4, 5, 5)
lr = SupportVectorMachineModel()
run = GeneticDataRun(data, "ciprofloxacin", bac, lr, selector)
print(run.run())
