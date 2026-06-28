from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INDEX_DATA_PATH, INDEX_PATH
from src.data import documents_from_cases
from src.embedding import LegalEmbedder


def build_index(
    cases: pd.DataFrame,
    embedder: LegalEmbedder,
    embedding_path: Path = INDEX_PATH,
    data_path: Path = INDEX_DATA_PATH,
) -> np.ndarray:
    embeddings = embedder.encode(documents_from_cases(cases))
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings)
    cases.to_csv(data_path, index=False, encoding="utf-8-sig")
    return embeddings


def load_index(
    embedding_path: Path = INDEX_PATH,
    data_path: Path = INDEX_DATA_PATH,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not embedding_path.exists() or not data_path.exists():
        raise FileNotFoundError("검색 인덱스가 없습니다. 먼저 인덱스를 생성하세요.")
    cases = pd.read_csv(data_path, encoding="utf-8-sig").fillna("")
    embeddings = np.load(embedding_path)
    if len(cases) != len(embeddings):
        raise ValueError("판례 데이터와 임베딩 개수가 일치하지 않습니다.")
    return cases, embeddings


def semantic_search(
    query: str,
    cases: pd.DataFrame,
    embeddings: np.ndarray,
    embedder: LegalEmbedder,
    top_k: int = 5,
) -> pd.DataFrame:
    if not query.strip():
        raise ValueError("검색할 사건 내용이나 쟁점을 입력하세요.")
    query_embedding = embedder.encode([query])[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(cases))]
    results = cases.iloc[top_indices].copy()
    results.insert(0, "similarity", scores[top_indices])
    return results.reset_index(drop=True)

