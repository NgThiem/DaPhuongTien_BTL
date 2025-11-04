import faiss
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import cv2
import torch
from torchvision.models import resnet18, ResNet18_Weights
from cnn_features import manual_transform
from sift_bovw import detectAndCompute

# ---------------- CONFIG ----------------
INDEX_FILE = Path("D:/DaPhuongTien/index.faiss")
PATHS_FILE = Path("D:/DaPhuongTien/paths.npy")
BOVW_KMEANS_FILE = Path("D:/DaPhuongTien/bovw_kmeans.pkl")
DEVICE = 'cpu'
K = 256
TOP_K = 5

# ---------------- Load index & paths ----------------
index = faiss.read_index(str(INDEX_FILE))
paths = np.load(PATHS_FILE, allow_pickle=True).tolist()

# ---------------- Load BoVW KMeans ----------------
kmeans = joblib.load(BOVW_KMEANS_FILE)

# ---------------- Load CNN model ----------------
cnn_model = resnet18(weights=ResNet18_Weights.DEFAULT)
cnn_model.fc = torch.nn.Identity()
cnn_model = cnn_model.to(DEVICE).eval()


# ---------------- Hàm tính vector kết hợp ----------------
def get_combined_vector(img_path):
    # ---------------- CNN vector ----------------
    img = Image.open(img_path).convert('RGB')
    x = manual_transform(img).unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        cnn_vec = cnn_model(x).cpu().numpy().flatten()
    
    # Tự tính L2 norm cho CNN vector
    norm_cnn = np.sqrt(np.sum(cnn_vec**2))
    if norm_cnn > 1e-6:
        cnn_vec = cnn_vec / norm_cnn

    # ---------------- BoVW vector ----------------
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    bovw_vec = np.zeros(K, dtype=np.float32)

    if img_gray is not None:
        kp, des = detectAndCompute(img_gray)
        if des is not None and len(des) > 0:
             # Gán từ descriptor sang cluster bằng kmeans
            words = kmeans.predict(des.astype(np.float32))
            # Tính histogram
            hist = np.zeros(K, dtype=np.float32)
            for w in words:
                hist[int(w)] += 1.0
            # Tự chuẩn hóa L2 histogram
            norm_hist = np.sqrt(np.sum(hist**2))
            if norm_hist > 1e-6:
                hist /= norm_hist
            bovw_vec = hist.astype(np.float32)

    # ---------------- Combine vector ----------------
    combined = np.concatenate([cnn_vec, bovw_vec], axis=0)  # vẫn dùng concat thư viện

    # chuẩn hóa vector kết hợp
    norm_combined = np.sqrt(np.sum(combined**2))
    if norm_combined > 1e-6:
        combined /= norm_combined

    return combined.reshape(1, -1).astype('float32')


# ---------------- Hàm tìm top-k ảnh tương tự ----------------
def query_image(img_path, top_k=TOP_K):
    vec = get_combined_vector(img_path)
    D, I = index.search(vec, top_k)
    results = [(paths[idx], float(D[0][i])) for i, idx in enumerate(I[0])]
    return results

