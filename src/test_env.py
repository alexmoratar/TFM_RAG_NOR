import sys
import pandas as pd
import numpy as np
import fitz
import faiss
import sentence_transformers
import gradio as gr
import yaml

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("PyMuPDF loaded:", fitz.__name__)
print("faiss version:", faiss.__version__)
print("sentence-transformers version:", sentence_transformers.__version__)
print("gradio version:", gr.__version__)
print("yaml version:", yaml.__version__)
print("✅ Entorno RAG funcionando desde script")
