from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_ROOT = PROJECT_ROOT / "models"
MODEL_DIR = MODEL_ROOT / "legal-sbert"
SELECTED_MODEL_FILE = MODEL_ROOT / "selected_model.txt"

SAMPLE_DATA_PATH = DATA_DIR / "sample_cases.csv"
SAMPLE_PAIR_PATH = DATA_DIR / "training_pairs.csv"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CASES_PATH = PROCESSED_DATA_DIR / "cases.csv"
PROCESSED_TRAIN_PAIRS_PATH = PROCESSED_DATA_DIR / "training_pairs.csv"
PROCESSED_VALID_CASES_PATH = PROCESSED_DATA_DIR / "validation_cases.csv"
PROCESSED_VALID_PAIRS_PATH = PROCESSED_DATA_DIR / "validation_pairs.csv"
INDEX_PATH = ARTIFACT_DIR / "case_embeddings.npy"
INDEX_DATA_PATH = ARTIFACT_DIR / "indexed_cases.csv"

BASE_MODEL_NAME = "jhgan/ko-sroberta-multitask"
REQUIRED_COLUMNS = {
    "id",
    "case_name",
    "court",
    "decision_date",
    "case_number",
    "category",
    "issues",
    "summary",
    "text",
    "source_url",
}


def default_case_data_path() -> Path:
    if PROCESSED_CASES_PATH.exists():
        return PROCESSED_CASES_PATH
    return SAMPLE_DATA_PATH


def default_pair_data_path() -> Path:
    if PROCESSED_TRAIN_PAIRS_PATH.exists():
        return PROCESSED_TRAIN_PAIRS_PATH
    return SAMPLE_PAIR_PATH
