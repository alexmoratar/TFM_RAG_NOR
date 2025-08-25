#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

from modules import utils
from modules import pdf_chunker
from modules import embedder
from modules import indexer
from modules import metadata

def main():
    parser = argparse.ArgumentParser(description="Procesa un PDF y lo integra en el corpus (chunks → embeddings → índices)")
    parser.add_argument("--pdf_path", type=str, required=True, help="Ruta al PDF a procesar")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo config.yaml")
    args = parser.parse_args()

    # 1. Config y logging
    cfg = utils.load_config(args.config)
    utils.ensure_dirs(cfg)
    utils.setup_logging(cfg["paths"]["results"])

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        logging.error(f"No existe el PDF: {pdf_path}")
        sys.exit(1)

    # 2. Hash e idempotencia
    pdf_hash = utils.sha256_file(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    logging.info(f"Procesando PDF: {pdf_name} | SHA256: {pdf_hash}")

    if metadata.already_ingested(cfg["paths"]["metadata"], pdf_hash):
        logging.info("Este PDF ya fue procesado anteriormente. Saliendo.")
        sys.exit(0)

    # 3. PDF → Chunks
    base_pdf = os.path.splitext(pdf_name)[0]
    chunks_out = os.path.join(cfg["paths"]["chunks"], f"chunks_{base_pdf}.json")
    chunks = pdf_chunker.pdf_to_chunks(
        pdf_path=pdf_path,
        output_json_path=chunks_out,
        chunk_size=cfg.get("chunking", {}).get("size", 300)
    )

    # 4. Embeddings
    texts = [c["text"] for c in chunks]
    models = {
        "minilm": cfg["models"]["minilm"],
        "mpnet": cfg["models"]["mpnet"]
    }
    batch_size = cfg.get("embedding", {}).get("batch_size", 32)
    normalize = bool(cfg.get("embedding", {}).get("normalize", True))
    device = cfg.get("embedding", {}).get("device", "cpu")

    emb_minilm_path = os.path.join(cfg["paths"]["embeddings"], f"embeddings_minilm_{base_pdf}.npy")
    emb_mpnet_path  = os.path.join(cfg["paths"]["embeddings"], f"embeddings_mpnet_{base_pdf}.npy")

    emb_minilm = embedder.generate_embeddings(texts, models["minilm"], batch_size, normalize, device, emb_minilm_path)
    emb_mpnet  = embedder.generate_embeddings(texts, models["mpnet"], batch_size, normalize, device, emb_mpnet_path)

    # 5. FAISS (incremental)
    faiss_minilm_path = os.path.join(cfg["paths"]["faiss"], "faiss_minilm.faiss")
    faiss_mpnet_path  = os.path.join(cfg["paths"]["faiss"], "faiss_mpnet.faiss")
    indexer.update_faiss_index(emb_minilm, faiss_minilm_path)
    indexer.update_faiss_index(emb_mpnet,  faiss_mpnet_path)

    # 6. BM25 (reconstrucción completa)
    bm25_path  = os.path.join(cfg["paths"]["bm25"], "bm25_index.pkl")
    texts_path = os.path.join(cfg["paths"]["bm25"], "texts.json")
    ids_path   = os.path.join(cfg["paths"]["bm25"], "ids.json")
    indexer.rebuild_bm25_from_chunks_dir(cfg["paths"]["chunks"], bm25_path, texts_path, ids_path)

    # 7. Metadatos
    metadata.save_pdf_metadata_per_doc(chunks, cfg["paths"]["metadata"], pdf_name, pdf_hash, models)
    metadata.update_global_ingestion_index(cfg["paths"]["metadata"], pdf_name, pdf_hash)

    # 8. QA
    n_chunks = len(chunks)
    if emb_minilm.shape[0] != n_chunks or emb_mpnet.shape[0] != n_chunks:
        logging.error("Error: n_chunks != n_embeddings")
        sys.exit(1)

    logging.info(f"OK: {pdf_name} procesado correctamente | chunks={n_chunks}")

if __name__ == "__main__":
    main()
