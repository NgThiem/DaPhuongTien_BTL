import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import joblib
from tqdm import tqdm

# ---------------- CONFIG ----------------
IMG_DIR = Path("D:/DaPhuongTien/dataset_cleaned")
K = 256
sift = cv2.SIFT_create()
BOVW_FILE = Path("D:/DaPhuongTien/bovw_vectors.pkl")
KMEANS_FILE = Path("D:/DaPhuongTien/bovw_kmeans.pkl")

# ---------------- Lấy image paths ----------------
image_paths = [p for p in IMG_DIR.rglob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
print(f"Tim thay {len(image_paths)} anh")

# ---------------- Extract SIFT descriptors ----------------
des_list = []
for path in tqdm(image_paths, desc="Extracting SIFT features"):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        des_list.append(des)

if len(des_list) == 0:
    raise ValueError("Khong co descriptor nao duoc tao.")

all_des = np.vstack(des_list)
print(f"Tong so descriptor: {all_des.shape}")

# ---------------- Train KMeans ----------------
print("Training KMeans...")
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000, verbose=1)
kmeans.fit(all_des)
print("KMeans training xong.")
joblib.dump(kmeans, KMEANS_FILE)
print(f"Da luu KMeans vao {KMEANS_FILE}")

# ---------------- Tạo BoVW vectors ----------------
def bovw_hist(des, kmeans):
    if des.dtype != np.float32:
        des = des.astype(np.float32)
    words = kmeans.predict(des)
    hist, _ = np.histogram(words, bins=np.arange(K+1))
    hist = hist.astype(float)
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist /= norm
    return hist

bovw_vectors = {}
for path in tqdm(image_paths, desc="Creating BoVW vectors"):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    kp, des = sift.detectAndCompute(img, None)
    key = str(path).replace('\\','/')  # Chuẩn hóa key giống CNN
    if des is not None:
        bovw_vectors[key] = bovw_hist(des, kmeans)
    else:
        bovw_vectors[key] = np.zeros(K, dtype=np.float32)

joblib.dump(bovw_vectors, BOVW_FILE)
print(f"Da luu BoVW vectors vao {BOVW_FILE}")
