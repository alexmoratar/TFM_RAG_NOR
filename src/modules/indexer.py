import os
import glob
import json
import faiss
import pickle
import logging
from rank_bm25 import BM25Okapi
import numpy as np

# FAISS

def update_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    Actualiza un índice FAISS existente añadiendo nuevos embeddings,
    o crea uno nuevo si no existe.
    """
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        if embeddings.shape[1] != index.d:
            raise ValueError(f"Dim mismatch: embeddings={embeddings.shape[1]} vs index.d={index.d}")
        index.add(embeddings)
        logging.info(f"Añadidos {embeddings.shape[0]} vectores a índice existente {index_path}")
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logging.info(f"Índice nuevo creado en {index_path} con {embeddings.shape[0]} vectores")
    faiss.write_index(index, index_path)
    logging.info(f"FAISS persistido en {index_path}")

# BM25

def rebuild_bm25_from_chunks_dir(chunks_dir: str, bm25_path: str, texts_path: str, ids_path: str):
    """
    Reconstruye un índice BM25 desde todos los chunks disponibles en `chunks_dir`.
    Guarda el objeto BM25 (pickle) y las listas texts/ids en JSON.
    """
    logging.info("Reconstruyendo BM25 desde todos los chunks")
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunks_*.json")))
    all_texts, all_ids = [], []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [c["text"] for c in data]
        ids = [f'{c["pdf"]}_{c["chunk_index"]}' for c in data]
        all_texts.extend(texts)
        all_ids.extend(ids)
    if not all_texts:
        raise ValueError("No se encontraron textos para construir BM25")

    # Tokenización simple acordada (str.split())
    tokenized = [t.split() for t in all_texts]
    bm25 = BM25Okapi(tokenized)

    # Guardar outputs
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)

    logging.info(f"BM25 guardado en {bm25_path} | docs: {len(all_texts)}")
