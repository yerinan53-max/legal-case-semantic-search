import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import default_case_data_path
from src.data import load_cases
from src.embedding import LegalEmbedder
from src.search import build_index


def main() -> None:
    data_path = default_case_data_path()
    cases = load_cases(data_path)
    print(f"데이터: {data_path}")
    embedder = LegalEmbedder()
    embeddings = build_index(cases, embedder)
    print(f"완료: 판례 {len(cases)}건, 임베딩 차원 {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
