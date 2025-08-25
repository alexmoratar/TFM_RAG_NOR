import os
import re
import json
import fitz  # PyMuPDF
import unicodedata
import logging

# Helpers internos

HEADER_DEFAULTS = {"EN", "NIST Privacy Framework"}

def _is_url(line: str) -> bool:
    return "http" in line or "www." in line

def _clean_weird_chars(line: str) -> str:
    # Normaliza Unicode y elimina caracteres de control
    line = unicodedata.normalize("NFC", line)
    return "".join(ch if (ord(ch) >= 32 and ord(ch) != 127) else " " for ch in line)

def _is_title(line: str) -> bool:
    return (line.isupper() or
            re.match(r"^[A-Z]\.\s", line) or
            re.match(r"^\d+(\.\d+)*", line))

# Función principal

def pdf_to_chunks(pdf_path: str, output_json_path: str, chunk_size: int = 300, headers_to_remove=None) -> list:
    """
    Procesa un PDF y lo divide en chunks de tamaño fijo.
    Devuelve la lista de chunks y los guarda en un archivo JSON.
    """
    headers_to_remove = headers_to_remove or HEADER_DEFAULTS
    logging.info(f"Cargando PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    logging.info(f"Páginas totales: {n_pages}")

    page_records = []
    for i in range(n_pages):
        page = doc.load_page(i)
        lines = [l.strip() for l in page.get_text().split("\n") if l.strip()]
        clean_lines = []
        titulo_interpretado = None
        for line in lines:
            if line in headers_to_remove or _is_url(line) or line.isdigit():
                continue
            line_clean = _clean_weird_chars(line)
            if _is_title(line_clean) and titulo_interpretado is None:
                titulo_interpretado = line_clean
            clean_lines.append(line_clean)
        text_final = "\n".join(clean_lines)
        if len(text_final.split()) < 20:
            continue
        page_records.append({
            "pdf": os.path.basename(pdf_path),
            "page_real": i + 1,
            "page_util": len(page_records) + 1,
            "titulo_interpretado": titulo_interpretado,
            "text_clean": text_final
        })
    logging.info(f"Páginas útiles procesadas: {len(page_records)}")

    chunks = []
    palabras, paginas_chunk, titulos_chunk = [], [], set()
    idx_chunk = 1
    for p in page_records:
        palabras_pagina = p["text_clean"].split()
        if p["titulo_interpretado"]:
            titulos_chunk.add(p["titulo_interpretado"])
        paginas_chunk.append(p["page_real"])
        for w in palabras_pagina:
            palabras.append(w)
            if len(palabras) == chunk_size:
                chunk_text = " ".join(palabras)
                chunks.append({
                    "pdf": os.path.basename(pdf_path),
                    "pages": paginas_chunk.copy(),
                    "titles": list(titulos_chunk),
                    "chunk_index": idx_chunk,
                    "text": chunk_text,
                    "n_words": len(palabras)
                })
                idx_chunk += 1
                palabras, paginas_chunk, titulos_chunk = [], [], set()
    if palabras:
        chunk_text = " ".join(palabras)
        chunks.append({
            "pdf": os.path.basename(pdf_path),
            "pages": paginas_chunk.copy(),
            "titles": list(titulos_chunk),
            "chunk_index": idx_chunk,
            "text": chunk_text,
            "n_words": len(palabras)
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logging.info(f"Chunks generados: {len(chunks)} | Guardado en {output_json_path}")
    return chunks
