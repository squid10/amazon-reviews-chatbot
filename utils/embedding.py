import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_modelo = None


def carregar_modelo(nome: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Carrega o SentenceTransformer uma única vez por processo."""
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(nome)
    return _modelo


def embed(textos, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Converte textos em vetores normalizados (o produto interno já é o cosseno)."""
    if isinstance(textos, str):        # sem isto, list("abc") viraria ["a", "b", "c"]
        textos = [textos]
    vetores = carregar_modelo().encode(
        list(textos),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vetores.astype("float32")
