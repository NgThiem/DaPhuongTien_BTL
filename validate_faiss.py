import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path (if query_image is defined there)
sys.path.append(str(Path(__file__).parent / "src"))
from query import query_image

# ---------------- CONFIG ----------------
VAL_DIR = Path("D:/DaPhuongTien/data/split/val")       # Query images
INDEX_DATASET = Path("D:/DaPhuongTien/dataset_cleaned")  # Indexed dataset
TOP_K = 10
# ---------------- Stats ----------------
total_queries = 0
correct_topk = 0
precision_list = []

# ---------------- Check data ----------------
if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

val_images = list(VAL_DIR.rglob("*.jpg")) + list(VAL_DIR.rglob("*.png")) + list(VAL_DIR.rglob("*.jpeg"))
print(f"Found {len(val_images)} validation images.")

if len(val_images) == 0:
    print("No images found in validation folder. Please check the path.")
    sys.exit()

# ---------------- Evaluation ----------------
for img_path in tqdm(val_images, desc="Evaluating"):
    total_queries += 1
    class_name = Path(img_path).parent.name

    try:
        top_results = query_image(str(img_path), TOP_K)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

    # Count same-class images in Top-K results
    correct = sum(1 for p, _ in top_results if Path(p).parent.name == class_name)

    if correct > 0:
        correct_topk += 1  # Top-K accuracy

    precision_list.append(correct / TOP_K)  # Precision@K

# ---------------- Metrics ----------------
if total_queries == 0:
    print("No valid queries for evaluation.")
else:
    topk_accuracy = correct_topk / total_queries
    mean_precision = np.mean(precision_list)

    print("--------------------------------------------------")
    print(f"Total query images: {total_queries}")
    print(f"Top-{TOP_K} Accuracy: {topk_accuracy:.4f}")
    print(f"Mean Precision@{TOP_K}: {mean_precision:.4f}")
    print("--------------------------------------------------")
