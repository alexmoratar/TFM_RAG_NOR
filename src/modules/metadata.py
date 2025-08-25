import os
import json
import logging
from modules.utils import now_iso

# Guardar metadatos por documento

def save_pdf_metadata_per_doc(chunks: list, metadata_dir: str, pdf_name: str, pdf_hash: str, models: dict):
    """
    Guarda un archivo JSON con los metadatos de todos los chunks de un PDF procesado.
    """
    out_path = os.path.join(metadata_dir, f"metadata_{os.path.splitext(pdf_name)[0]}.json")
    date_idx = now_iso()
    enriched = []
    for c in chunks:
        enriched.append({
            "chunk_id": c["chunk_index"],
            "pdf": c["pdf"],
            "chunk_index": c["chunk_index"],
            "pages": c.get("pages", []),
            "title": (c.get("titles") or [None])[0] if c.get("titles") else None,
            "text": c["text"],
            "hash_pdf": f"sha256:{pdf_hash}",
            "embedding_models": models,
            "date_indexed": date_idx
        })
    os.makedirs(metadata_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    logging.info(f"Metadatos por documento guardados en {out_path}")
    return out_path

# Índice global de ingestas

def update_global_ingestion_index(metadata_dir: str, pdf_name: str, pdf_hash: str):
    """
    Actualiza el archivo ingestion_index.json con la info del PDF procesado.
    Evita duplicados por hash.
    """
    idx_path = os.path.join(metadata_dir, "ingestion_index.json")
    record = {
        "pdf": pdf_name,
        "sha256": pdf_hash,
        "date_indexed": now_iso()
    }
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
    else:
        arr = []
    if any(r.get("sha256") == pdf_hash for r in arr):
        logging.info("El PDF ya estaba registrado en el índice global (hash duplicado).")
        return idx_path
    arr.append(record)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    logging.info(f"Ingestion index actualizado en {idx_path}")
    return idx_path

def already_ingested(metadata_dir: str, pdf_hash: str) -> bool:
    """
    Comprueba si un PDF ya fue procesado, buscando su hash en ingestion_index.json.
    """
    idx_path = os.path.join(metadata_dir, "ingestion_index.json")
    if not os.path.exists(idx_path):
        return False
    with open(idx_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return any(r.get("sha256") == pdf_hash for r in arr)

