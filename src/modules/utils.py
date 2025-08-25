import os
import sys
import yaml
import hashlib
import logging
from datetime import datetime

# Configuración y logging

def load_config(path: str) -> dict:
    """Carga un archivo config.yaml y lo devuelve como diccionario"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(cfg: dict):
    """Crea todas las carpetas necesarias definidas en config.yaml"""
    for k in ("chunks", "embeddings", "faiss", "bm25", "metadata", "results"):
        os.makedirs(cfg["paths"][k], exist_ok=True)

def setup_logging(results_dir: str):
    """Inicializa logging en consola y en archivo process.log"""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "process.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging inicializado")

# Funciones auxiliares

def sha256_file(path: str) -> str:
    """Devuelve el hash SHA256 de un archivo"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def now_iso() -> str:
    """Devuelve la fecha actual en formato YYYY-MM-DD"""
    return datetime.now().strftime("%Y-%m-%d")