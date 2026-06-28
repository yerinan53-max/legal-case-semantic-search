import pandas as pd
import pytest

from src.analysis import category_counts, extract_issue_keywords
from src.data import build_document_text, load_cases


def test_load_sample_cases(tmp_path):
    path = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "id": "1",
                "case_name": "테스트 사건",
                "court": "테스트 법원",
                "decision_date": "2025-01-01",
                "case_number": "2025테스트1",
                "category": "민사",
                "issues": "계약 해제",
                "summary": "계약 해제 요건을 판단하였다.",
                "text": "채무불이행을 이유로 계약 해제를 청구하였다.",
                "source_url": "",
            }
        ]
    ).to_csv(path, index=False, encoding="utf-8")

    cases = load_cases(path)
    assert len(cases) == 1
    assert "계약 해제" in build_document_text(cases.iloc[0])


def test_missing_columns_raise_error(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame([{"id": "1"}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_cases(path)


def test_issue_analysis():
    results = pd.DataFrame(
        {
            "issues": ["계약 해제 손해배상", "계약 무효 손해배상"],
            "summary": ["채무불이행", "의사표시"],
            "category": ["민사", "민사"],
        }
    )
    assert extract_issue_keywords(results)[0][0] in {"계약", "손해배상"}
    assert category_counts(results).iloc[0]["count"] == 2

