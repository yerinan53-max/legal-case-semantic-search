import numpy as np
import pandas as pd

from src.search import semantic_search


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0]], dtype="float32")


def test_exact_legal_term_reranks_close_semantic_results():
    cases = pd.DataFrame(
        [
            {
                "case_name": "중상해",
                "issues": "상해의 고의",
                "summary": "피해자에게 중한 상해가 발생하였다.",
            },
            {
                "case_name": "폭행",
                "issues": "폭행",
                "summary": "사람의 신체에 유형력을 행사하였다.",
            },
        ]
    )
    embeddings = np.array([[0.58], [0.56]], dtype="float32")

    results = semantic_search(
        "행인을 묻지마 폭행함",
        cases,
        embeddings,
        FakeEmbedder(),
        top_k=2,
    )

    assert results.iloc[0]["case_name"] == "폭행"
    assert results.iloc[0]["semantic_similarity"] == np.float32(0.56)
