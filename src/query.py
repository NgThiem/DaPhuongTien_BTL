import faiss
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

sift = cv2.SIFT_create()

# ---------------- Hàm tính vector kết hợp ----------------
def get_combined_vector(img_path):
    # --- CNN vector ---
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cnn_vec = cnn_model(x).cpu().numpy().flatten()
    cnn_vec /= np.linalg.norm(cnn_vec)+1e-6

    # --- BoVW vector ---
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        bovw_vec = np.zeros(K, dtype=np.float32)
    else:
        kp, des = sift.detectAndCompute(img_gray, None)
        if des is None:
            bovw_vec = np.zeros(K, dtype=np.float32)
        else:
            words = kmeans.predict(des.astype(np.float32))
            hist, _ = np.histogram(words, bins=np.arange(K+1))
            hist = hist.astype(float)
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist /= norm
            bovw_vec = hist.astype(np.float32)

    # --- Combine ---
    combined = np.concatenate([cnn_vec, bovw_vec], axis=0)
    combined /= np.linalg.norm(combined)+1e-6
    return combined.reshape(1,-1).astype('float32')

# ---------------- Hàm tìm top-k ảnh tương tự ----------------
def query_image(img_path, top_k=TOP_K):
    vec = get_combined_vector(img_path)
    D, I = index.search(vec, top_k)
    results = [(paths[idx], float(D[0][i])) for i, idx in enumerate(I[0])]
    return results

# ---------------- Ví dụ test ----------------
if __name__ == "__main__":
    query_img = "D:/DaPhuongTien/data/split/test/tao/001.jpg"
    top_results = query_image(query_img, TOP_K)
    print(f"Top {TOP_K} anh tuong tự:")
    for path, score in top_results:
        print(f"{path} (score={score:.4f})")
