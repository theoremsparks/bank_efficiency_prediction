from pathlib import Path

# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Main folders
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "analysis"
ML_OUTPUT_DIR = OUTPUTS_DIR / "ml"

MODELS_DIR = BASE_DIR / "models"
ML_MODELS_DIR = MODELS_DIR / "ml"

# Main files
RAW_DATA_PATH = RAW_DATA_DIR / "data.csv"
DEA_PROCESSED_PATH = PROCESSED_DATA_DIR / "dea_data_for_ml.csv"

# Create folders automatically if they do not exist
for folder in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    ANALYSIS_OUTPUT_DIR,
    ML_OUTPUT_DIR,
    MODELS_DIR,
    ML_MODELS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


def get_analysis_output_dir(name: str):
    path = ANALYSIS_OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_output_dir(name: str):
    path = ML_OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_dir(name: str):
    path = ML_MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path