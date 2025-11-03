from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import faiss
import sys

sys.path.append(str(Path(__file__).parent / "src"))
from cnn_features import manual_transform
from sift_bovw import bovw_hist

# ---------------- CONFIG ----------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset_cleaned'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

INDEX_FILE = "index.faiss"
PATHS_FILE = "paths.npy"
KMEANS_FILE = "bovw_kmeans.pkl"
VECTORS_FILE = "bovw_vectors.pkl"
TOP_K_DEFAULT = 5

CNN_WEIGHT = 0.9
BOVW_WEIGHT = 0.1
K = 256  # số cluster BoVW
SIFT_STEP = 8  # step khi quét ảnh SIFT
RESIZE_DIM = (224, 224)  # resize ảnh trước khi tính CNN / SIFT

# ---------------- LOAD MODELS / FILES ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load FAISS index và paths
index = faiss.read_index(INDEX_FILE)
paths = np.load(PATHS_FILE, allow_pickle=True).tolist()

# Load KMeans và BoVW vectors
bovw_kmeans = joblib.load(KMEANS_FILE)
bovw_vectors = joblib.load(VECTORS_FILE)

# Load CNN model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(device).float().eval()

# ---------------- CACHE ----------------
cnn_vectors_cache = {}
bovw_vectors_cache = {}

# ---------------- FEATURE EXTRACTION ----------------
def extract_cnn_features(img_path):
    key = str(img_path).replace('\\','/')
    if key in cnn_vectors_cache:
        return cnn_vectors_cache[key]

    try:
        img = Image.open(img_path).convert('RGB').resize(RESIZE_DIM)
    except Exception as e:
        print(f"Warning: failed to open {img_path} -> {e}")
        vec = np.zeros(512, dtype=np.float32)
        cnn_vectors_cache[key] = vec
        return vec

    x = manual_transform(img).unsqueeze(0).to(device).float()
    with torch.no_grad():
        vec = model(x).squeeze().cpu().numpy().astype(np.float32)
    vec /= (np.linalg.norm(vec)+1e-6)
    cnn_vectors_cache[key] = vec
    return vec

def extract_bovw_features(img_path):
    key = str(img_path).replace('\\','/')
    if key in bovw_vectors_cache:
        return bovw_vectors_cache[key]

    if key in bovw_vectors:
        hist = bovw_vectors[key]
        bovw_vectors_cache[key] = hist
        return hist

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Warning: failed to read {img_path}")
        hist = np.zeros(K, dtype=np.float32)
        bovw_vectors_cache[key] = hist
        return hist

    gray = cv2.resize(gray, RESIZE_DIM)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        hist = np.zeros(K, dtype=np.float32)
        bovw_vectors_cache[key] = hist
        return hist

    centroids = bovw_kmeans.cluster_centers_
    hist = bovw_hist(des, centroids)
    hist /= (np.linalg.norm(hist)+1e-6)

    bovw_vectors_cache[key] = hist
    return hist

def get_combined_vector(img_path):
    cnn_vec = extract_cnn_features(img_path)
    bovw_vec = extract_bovw_features(img_path)
    combined = np.concatenate([cnn_vec * CNN_WEIGHT, bovw_vec * BOVW_WEIGHT])
    combined /= (np.linalg.norm(combined)+1e-6)
    return combined.astype('float32').reshape(1, -1)

# ---------------- SEARCH ----------------
def retrieve_top_k(query_path, k=TOP_K_DEFAULT):
    query_vec = get_combined_vector(query_path)
    D, I = index.search(query_vec, k)
    results = [(paths[idx], float(D[0][i])) for i, idx in enumerate(I[0])]
    return results

# ---------------- ROUTES ----------------
@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    top_k = int(request.form.get('top_k', TOP_K_DEFAULT))
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    results = retrieve_top_k(filepath, k=top_k)
    top_urls = [{"url": f"/dataset/{os.path.relpath(p, DATASET_FOLDER).replace('\\','/')}", 
                 "rel_path": os.path.relpath(p, DATASET_FOLDER).replace('\\','/')} 
                for p, score in results]

    query_url = f"/uploads/{file.filename}"
    return jsonify({"query": query_url, "results": top_urls})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/dataset/<path:filename>')
def dataset(filename):
    return send_from_directory(DATASET_FOLDER, filename)

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    query_rel = os.path.basename(data.get('query_path'))
    query_path = os.path.join(UPLOAD_FOLDER, query_rel)
    result_rel = data.get('result_rel')
    result_path = os.path.join(DATASET_FOLDER, result_rel)

    q_feat = get_combined_vector(query_path)
    r_feat = get_combined_vector(result_path)
    sim = float(np.dot(q_feat, r_feat.T)[0][0])
    sim_percent = round(sim * 100, 2)
    return jsonify({'similarity': sim_percent})

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
