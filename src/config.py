from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models" / "legal-sbert"

SAMPLE_DATA_PATH = DATA_DIR / "sample_cases.csv"
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

