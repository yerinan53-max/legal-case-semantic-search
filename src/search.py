from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.config import INDEX_DATA_PATH, INDEX_PATH
from src.data import documents_from_cases
from src.embedding import LegalEmbedder


def expand_legal_query(query: str) -> str:
    normalized = _normalize_text(query)
    expansions = []

    vehicle_sale_terms = ("중고차", "무사고", "사고이력")
    transaction_terms = ("구매", "매매", "판매", "샀", "구입")
    if any(term in normalized for term in vehicle_sale_terms) and any(
        term in normalized for term in transaction_terms
    ):
        expansions.append(
            "자동차매매 사고이력 미고지 기망 고지의무 하자담보책임 "
            "계약취소 매매대금반환 사기"
        )

    if any(term in normalized for term in ("전세금", "보증금")) and any(
        term in normalized for term in ("임대차", "집주인", "임대인", "반환")
    ):
        expansions.append(
            "임대차보증금 전세금 반환의무 계약종료 임차인 임대인"
        )

    if "해고" in normalized:
        expansions.append("부당해고 징계해고 해고의 정당성 징계양정")

    if "개인정보" in normalized:
        expansions.append("개인정보 제3자 제공 동의 개인정보보호법 손해배상")

    return " ".join([query, *expansions])


def _normalize_text(text: object) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", str(text)).lower()


def _lexical_scores(
    query: str,
    cases: pd.DataFrame,
    title_query: str | None = None,
) -> np.ndarray:
    query_normalized = _normalize_text(title_query or query)
    query_terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", query.lower())
    scores = []

    for _, row in cases.iterrows():
        title = _normalize_text(row.get("case_name", ""))
        searchable = _normalize_text(
            f"{row.get('case_name', '')} {row.get('issues', '')} "
            f"{row.get('summary', '')}"
        )
        exact_title = bool(title and title in query_normalized)
        term_overlap = (
            sum(_normalize_text(term) in searchable for term in query_terms)
            / len(query_terms)
            if query_terms
            else 0.0
        )
        scores.append(min(1.0, 0.7 * float(exact_title) + 0.3 * term_overlap))

    return np.asarray(scores, dtype="float32")


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
    expanded_query = expand_legal_query(query)
    query_embedding = embedder.encode([expanded_query])[0]
    semantic_scores = embeddings @ query_embedding
    lexical_scores = _lexical_scores(expanded_query, cases, title_query=query)
    scores = 0.9 * semantic_scores + 0.1 * lexical_scores
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(cases))]
    results = cases.iloc[top_indices].copy()
    results.insert(0, "similarity", scores[top_indices])
    results.insert(1, "semantic_similarity", semantic_scores[top_indices])
    return results.reset_index(drop=True)
