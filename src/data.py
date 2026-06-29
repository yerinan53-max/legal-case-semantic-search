from pathlib import Path

import pandas as pd

from src.config import REQUIRED_COLUMNS


def load_cases(path_or_buffer: str | Path | object) -> pd.DataFrame:
    """Load and validate a case CSV."""
    cases = pd.read_csv(path_or_buffer, encoding="utf-8")
    missing = REQUIRED_COLUMNS.difference(cases.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")

    cases = cases.copy()
    for column in REQUIRED_COLUMNS:
        cases[column] = cases[column].fillna("").astype(str).str.strip()
    cases = cases[cases["text"].ne("") | cases["summary"].ne("")]
    if cases.empty:
        raise ValueError("검색에 사용할 판례 내용이 없습니다.")
    if cases["id"].duplicated().any():
        raise ValueError("id 열에는 중복되지 않는 값이 필요합니다.")
    return cases.reset_index(drop=True)


def build_document_text(row: pd.Series) -> str:
    """Combine searchable legal fields into one embedding document."""
    issues = row["issues"][:500]
    summary = row["summary"][:1_200]
    body = row["text"][:800]
    return (
        f"사건명: {row['case_name']}\n"
        f"분야: {row['category']}\n"
        f"쟁점: {issues}\n"
        f"판결요지: {summary}\n"
        f"본문: {body}"
    )


def documents_from_cases(cases: pd.DataFrame) -> list[str]:
    return [build_document_text(row) for _, row in cases.iterrows()]
