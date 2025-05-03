from sklearn.model_selection import train_test_split

from constants.constants import ROOT_DIR
from data_providers.genetic_data.genetic_data import GeneticDataset
from feature_selection.banzaff_power_index import BanzhafFeatureSelector
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
    "ciprofloxacin",
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
data = GeneticDataset(bac_name=bac, max_sra_ids=30, root_dir='/mnt/d')
selector = BanzhafFeatureSelector(150, 4, 5, 5)
lr = SupportVectorMachineModel()
run = GeneticDataRun(data, "ciprofloxacin", bac, lr, selector)
print(run.run())
