import os
import time
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# Embeddings

def generate_embeddings(
    texts: list,
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
    device: str = "cpu",
    out_path: str = None
) -> np.ndarray:
    """
    Genera embeddings para una lista de textos usando un modelo de SentenceTransformers.
    Devuelve un array numpy y opcionalmente lo guarda en un archivo .npy
    """
    logging.info(f"Generando embeddings con {model_name} | batch={batch_size} | normalize={normalize} | device={device}")
    model = SentenceTransformer(model_name, device=device)

    start = time.time()
    embs = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=normalize
    )
    elapsed = time.time() - start
    logging.info(f"Embeddings generados | dim={embs.shape} | tiempo={elapsed:.2f}s")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, embs)
        logging.info(f"Embeddings guardados en {out_path}")

    return embs

def save_embedding_metadata(model_name: str, n_chunks: int, elapsed: float, out_path: str):
    """
    Guarda un JSON con información sobre el proceso de embeddings
    """
    result = {
        "modelo": model_name,
        "n_chunks": n_chunks,
        "tiempo_segundos": round(elapsed, 2)
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Metadata de embeddings guardada en {out_path}")
