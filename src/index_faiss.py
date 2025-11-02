import faiss
import numpy as np
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path("D:/DaPhuongTien")
CNN_FILE = PROJECT_ROOT / "cnn_vectors.pkl"
BOVW_FILE = PROJECT_ROOT / "bovw_vectors.pkl"
INDEX_FILE = PROJECT_ROOT / "index.faiss"
PATHS_FILE = PROJECT_ROOT / "paths.npy"

# ---------------- Load vectors ----------------
cnn_vectors = joblib.load(CNN_FILE)
bovw_vectors = joblib.load(BOVW_FILE)
print(f"Da load {len(cnn_vectors)} CNN vectors va {len(bovw_vectors)} BoVW vectors")

# ---------------- Chuẩn hóa key ----------------
common_keys = sorted(set(cnn_vectors.keys()) & set(bovw_vectors.keys()))
print(f"So anh trung khop: {len(common_keys)}")

cnn_mat = np.stack([cnn_vectors[k] for k in common_keys]).astype('float32')
bovw_mat = np.stack([bovw_vectors[k] for k in common_keys]).astype('float32')
combined_mat = np.concatenate((cnn_mat, bovw_mat), axis=1).astype('float32')

# ---------------- Tạo Faiss index ----------------
vector_dim = combined_mat.shape[1]
faiss.normalize_L2(combined_mat)  # chuẩn hóa cho cosine similarity
index = faiss.IndexFlatIP(vector_dim)
index.add(combined_mat)
print(f"Da them {index.ntotal} vector vao Faiss Index")

# ---------------- Lưu index và paths ----------------
faiss.write_index(index, str(INDEX_FILE))
np.save(PATHS_FILE, np.array(common_keys))
print(f"Da luu index Faiss vao {INDEX_FILE} va paths vao {PATHS_FILE}")
