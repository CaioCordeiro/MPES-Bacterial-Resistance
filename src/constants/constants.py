ROOT_DIR = "."

SRA_ID_LIST_KLEB = ["SRR5386221", "SRR5386094", "SRR5386629"]
SRA_ID_LIST_ECH = [
    "ERR2091428",
    "ERR2091335",
    "ERR2931009",
    "ERR2091427",
    "ERR2091312",
    "ERR2091418",
    "ERR3808903",
    "ERR3808914",
    "ERR3808888",
    "ERR2931033",
    "ERR3808921",
    "ERR2091340",
    "ERR2091308",
    "ERR2091307",
    "ERR2091328",
    "ERR3808887",
    "ERR2091316",
    "ERR3808890",
    "ERR2091339",
    "ERR3808876",
    "ERR2091319",
    "ERR2091414",
    "ERR2091312",
    "ERR2091397",
    "ERR3808908",
    "ERR2091401",
]
EXCLUDE_LIST = []
K_SIZE = 5
FILE_DIR = "data/raw_data"
OUTPUT_DIR = "data/features"
ANTIBIOTIC_LIST = [
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
ANTIBIOTIC_FILE = "data/anti_rel"
FEATURE_DIR = "data/features"
DATASET_OUTPUT_DIR = "data/datasets"

TOTAL_ROW_NUMBER = 50

# Resource limits
MAX_MEMORY_PERCENT = 80.0
MAX_CONCURRENT_DOWNLOADS = 3
MAX_PROCESSES = 4

CACHE_DIR = "data/cache"
