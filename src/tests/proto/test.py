import faiss
import os
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]

faiss_index_path = "faiss_index.bin"

print(f"Attempting to read Faiss index from: {faiss_index_path}\n")

# Check if the file exists before attempting to read it
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"Faiss index file not found at: {faiss_index_path}")

# Read the Faiss index
try:
    faiss_index = faiss.read_index(faiss_index_path)
except Exception as e:
    print(f"Failed to read Faiss index: {e}")
    raise