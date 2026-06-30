from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import documents_from_cases, load_cases


VAL_CASES_PATH = ROOT / "data" / "processed" / "validation_cases.csv"
VAL_PAIRS_PATH = ROOT / "data" / "processed" / "validation_pairs.csv"
MODEL_PATH = ROOT / "models" / "legal-sbert"
OUTPUT_DIR = ROOT / "presentation" / "assets" / "experiments"
JSON_PATH = OUTPUT_DIR / "embedding_similarity_heatmap.json"
PNG_PATH = OUTPUT_DIR / "embedding_similarity_heatmap.png"
SAMPLE_SIZE = 10
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"


def main() -> None:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    cases = load_cases(VAL_CASES_PATH)
    cases["id"] = cases["id"].astype(str)
    pairs = pd.read_csv(VAL_PAIRS_PATH)
    pairs["case_id"] = pairs["case_id"].astype(str)

    valid_ids = set(cases["id"])
    selected = (
        pairs[pairs["case_id"].isin(valid_ids)]
        .drop_duplicates("case_id")
        .sample(n=SAMPLE_SIZE, random_state=42)
        .reset_index(drop=True)
    )
    case_lookup = cases.set_index("id")
    selected_cases = case_lookup.loc[selected["case_id"]].reset_index()

    model = SentenceTransformer(str(MODEL_PATH))
    query_embeddings = model.encode(
        selected["anchor"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    document_embeddings = model.encode(
        documents_from_cases(selected_cases),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = query_embeddings @ document_embeddings.T

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "Legal Ko-SRoBERTa (1 Epoch)",
        "sample_size": SAMPLE_SIZE,
        "query_labels": [f"Q{i}" for i in range(1, SAMPLE_SIZE + 1)],
        "case_labels": [f"C{i}" for i in range(1, SAMPLE_SIZE + 1)],
        "values": similarities.round(4).tolist(),
        "diagonal_mean": float(np.diag(similarities).mean()),
        "off_diagonal_mean": float(
            similarities[~np.eye(SAMPLE_SIZE, dtype=bool)].mean()
        ),
    }
    JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(similarities, cmap="YlGnBu", vmin=-0.2, vmax=1.0)
    for row in range(SAMPLE_SIZE):
        for column in range(SAMPLE_SIZE):
            value = similarities[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "#172A46",
            )
    ax.set_xticks(range(SAMPLE_SIZE), payload["case_labels"])
    ax.set_yticks(range(SAMPLE_SIZE), payload["query_labels"])
    ax.set_xlabel("정답 후보 판례")
    ax.set_ylabel("검증 질의")
    ax.set_title("질의–판례 임베딩 코사인 유사도")
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=180)
    plt.close(fig)
    print(payload)


if __name__ == "__main__":
    main()
