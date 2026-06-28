from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import BASE_MODEL_NAME, MODEL_DIR


def resolve_model_name() -> str:
    return str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else BASE_MODEL_NAME


class LegalEmbedder:
    def __init__(self, model_name: str | Path | None = None) -> None:
        self.model_name = str(model_name or resolve_model_name())
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

