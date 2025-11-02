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
BATCH_SIZE = 32
SAVE_EVERY = 500
IMG_DIR = Path("D:/DaPhuongTien/dataset_cleaned")
CNN_VECTORS_FILE = Path("D:/DaPhuongTien/cnn_vectors.pkl")

# --------------------- Model CNN ---------------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(DEVICE).eval()

# --------------------- Transform ảnh ---------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

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
    imgs = [transform(Image.open(p).convert('RGB')) for p in paths]
    return torch.stack(imgs)

for i in range(0, len(all_paths), BATCH_SIZE):
    batch_paths = all_paths[i:i+BATCH_SIZE]
    # Chuẩn hóa key
    batch_paths = [p for p in batch_paths if str(p).replace('\\','/') not in cnn_vectors]
    if not batch_paths:
        continue
    batch_imgs = load_images(batch_paths).to(DEVICE)
    with torch.no_grad():
        batch_vec = model(batch_imgs).cpu().numpy()
    for p, v in zip(batch_paths, batch_vec):
        key = str(p).replace('\\','/')
        cnn_vectors[key] = v / (np.linalg.norm(v)+1e-6)
    if len(cnn_vectors) % SAVE_EVERY < BATCH_SIZE:
        joblib.dump(cnn_vectors, CNN_VECTORS_FILE)
        print(f"Da luu tam {len(cnn_vectors)} vector CNN")

joblib.dump(cnn_vectors, CNN_VECTORS_FILE)
print(f"Da luu cuoi cung {len(cnn_vectors)} vector CNN vao {CNN_VECTORS_FILE}")
