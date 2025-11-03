import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import joblib
from tqdm import tqdm

# --------------------- Cấu hình ---------------------
DEVICE = 'cpu'  # hoặc 'cuda' nếu có GPU
BATCH_SIZE = 32 #số ảnh xử lý 1 lần.
SAVE_EVERY = 500
IMG_DIR = Path("D:/DaPhuongTien/dataset_cleaned")
CNN_VECTORS_FILE = Path("D:/DaPhuongTien/cnn_vectors.pkl")

# --------------------- Model CNN ---------------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(DEVICE).eval()

# --------------------- Transform ảnh ---------------------
def manual_transform(img_pil):
    """Thay cho transforms.Compose(...)"""
    # Resize về 256, crop trung tâm 224x224
    img_pil = img_pil.resize((256, 256))
    w, h = img_pil.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    img_pil = img_pil.crop((left, top, left + 224, top + 224))

    # Chuyển sang numpy array và chuẩn hóa giá trị [0,1]
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    # Nếu ảnh không đủ 3 kênh, lặp lại cho đủ RGB
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    # Chuẩn hóa theo mean/std của ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std

    # Chuyển sang tensor (C,H,W)
    img_tensor = torch.tensor(img_np.transpose(2, 0, 1))
    return img_tensor

# --------------------- Load danh sách ảnh ---------------------
all_paths = [p for p in IMG_DIR.rglob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
print(f"Tim thay {len(all_paths)} anh")

# --------------------- Resume nếu đã có file ---------------------
if CNN_VECTORS_FILE.exists():
    cnn_vectors = joblib.load(CNN_VECTORS_FILE)
    print(f"Da load {len(cnn_vectors)} vector CNN truoc do")
else:
    cnn_vectors = {}

# --------------------- Batch extract ---------------------
def load_images(paths):
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            img_tensor = manual_transform(img)
            imgs.append(img_tensor)
        except Exception as e:
            print(f"Lỗi khi đọc {p}: {e}")
    if not imgs:
        return torch.empty(0)
    return torch.stack(imgs).float()

# Trích xuất đặc trưng

for i in range(0, len(all_paths), BATCH_SIZE):
    batch_paths = all_paths[i:i+BATCH_SIZE]
    # Chuẩn hóa key
    batch_paths = [p for p in batch_paths if str(p).replace('\\','/') not in cnn_vectors]
    if not batch_paths:
        continue
    batch_imgs = load_images(batch_paths).to(DEVICE)
    with torch.no_grad():
        batch_vec = model(batch_imgs.float()).cpu().numpy()
        
# Chuẩn hóa vector
    for p, v in zip(batch_paths, batch_vec):
        norm = np.sqrt(np.sum(v ** 2))
        if norm > 1e-6:
            v = v / norm
        key = str(p).replace('\\','/')
        cnn_vectors[key] = v

    if len(cnn_vectors) % SAVE_EVERY < BATCH_SIZE:
        joblib.dump(cnn_vectors, CNN_VECTORS_FILE)
        print(f"Da luu tam {len(cnn_vectors)} vector CNN")

joblib.dump(cnn_vectors, CNN_VECTORS_FILE)
print(f"Da luu cuoi cung {len(cnn_vectors)} vector CNN vao {CNN_VECTORS_FILE}")
