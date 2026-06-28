import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SAMPLE_DATA_PATH
from src.data import load_cases
from src.embedding import LegalEmbedder
from src.search import build_index


def main() -> None:
    cases = load_cases(SAMPLE_DATA_PATH)
    embedder = LegalEmbedder()
    embeddings = build_index(cases, embedder)
    print(f"완료: 판례 {len(cases)}건, 임베딩 차원 {embeddings.shape[1]}")


if __name__ == "__main__":
    main()

