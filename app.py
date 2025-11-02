from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import faiss

# ---------------- CONFIG ----------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset_cleaned'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

INDEX_FILE = "index.faiss"
PATHS_FILE = "paths.npy"
KMEANS_FILE = "bovw_kmeans.pkl"
TOP_K_DEFAULT = 5

CNN_WEIGHT = 0.9
BOVW_WEIGHT = 0.1
K = 256

# ---------------- LOAD MODELS ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

index = faiss.read_index(INDEX_FILE)
paths = np.load(PATHS_FILE, allow_pickle=True).tolist()
kmeans = joblib.load(KMEANS_FILE)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(device).eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

sift = cv2.SIFT_create()

# ---------------- FEATURE EXTRACTION ----------------
def extract_cnn_features(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(x).squeeze().cpu().numpy()
    vec /= np.linalg.norm(vec) + 1e-6
    return vec

def extract_bovw_features(img_path):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros(K, dtype=np.float32)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        return np.zeros(K, dtype=np.float32)
    words = kmeans.predict(des.astype(np.float32))
    hist, _ = np.histogram(words, bins=np.arange(K + 1))
    hist = hist.astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-6
    return hist

def get_combined_vector(img_path):
    cnn_vec = extract_cnn_features(img_path)
    bovw_vec = extract_bovw_features(img_path)
    combined = np.concatenate([cnn_vec * CNN_WEIGHT, bovw_vec * BOVW_WEIGHT])
    combined /= np.linalg.norm(combined) + 1e-6
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
    sim = np.dot(q_feat, r_feat.T)[0][0]
    sim_percent = round(sim * 100, 2)
    return jsonify({'similarity': float(sim_percent)})


if __name__ == '__main__':
    app.run(debug=True)
