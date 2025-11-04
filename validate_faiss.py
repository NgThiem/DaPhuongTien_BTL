from pathlib import Path
from tqdm import tqdm
import os
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from query import query_image
# ---------------- CONFIG ----------------
VAL_DIR = "D:/DaPhuongTien/data/split/val"
DATASET_DIR = "D:/DaPhuongTien/dataset_cleaned"
TOP_K = 10

# ---------------- Load dataset ----------------
dataset_paths = []
dataset_classes = []
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            dataset_paths.append(os.path.join(root,f))
            dataset_classes.append(os.path.basename(root))

dataset_dict = {p.replace("\\","/"): c for p,c in zip(dataset_paths, dataset_classes)}
classes = sorted(set(dataset_classes))

# ---------------- Load validation images ----------------
val_images = []
for root, dirs, files in os.walk(VAL_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            val_images.append(os.path.join(root,f))

# ---------------- Stats ----------------
per_class_stats = {c: {"tp":0, "fp":0, "fn":0} for c in classes}
topk_precision_list = []
topk_recall_list = []

# ---------------- Evaluation ----------------
for img_path in tqdm(val_images, desc="Evaluating"):
    query_class = os.path.basename(os.path.dirname(img_path))
    top_results = query_image(img_path, TOP_K)
    retrieved_classes = [dataset_dict.get(res_path.replace("\\","/"), None) for res_path,_ in top_results]

    # ---------------- Top-K precision/recall ----------------
    correct_in_topk = sum(1 for rc in retrieved_classes if rc == query_class)
    topk_precision_list.append(correct_in_topk / TOP_K)
    
    # Recall@K = TP trong top-K / tổng số ảnh cùng lớp trong dataset (trừ query)
    total_same_class = sum(1 for c in dataset_classes if c == query_class) - 1
    recall_k = correct_in_topk / total_same_class if total_same_class > 0 else 0
    topk_recall_list.append(recall_k)

    # ---------------- Update TP/FP/FN cho Micro/Macro F1 ----------------
    for c in classes:
        tp = sum(1 for rc in retrieved_classes if rc == c) if c == query_class else 0
        fp = sum(1 for rc in retrieved_classes if rc == c) if c != query_class else 0
        fn = sum(1 for p,c2 in zip(dataset_paths,dataset_classes) if c2 == c and c2 == query_class) - tp if c == query_class else 0

        per_class_stats[c]["tp"] += tp
        per_class_stats[c]["fp"] += fp
        per_class_stats[c]["fn"] += fn

# ---------------- Metrics ----------------
f1_macro_list = []
tp_micro = fp_micro = fn_micro = 0

for c, stats in per_class_stats.items():
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    f1_macro_list.append(f1)

    tp_micro += tp
    fp_micro += fp
    fn_micro += fn

f1_macro = sum(f1_macro_list) / len(f1_macro_list)
precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
f1_micro = 2*precision_micro*recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

topk_precision_mean = sum(topk_precision_list) / len(topk_precision_list)
topk_recall_mean = sum(topk_recall_list) / len(topk_recall_list)

print("--------------------------------------------------")
print(f"Top-{TOP_K} Precision@K: {topk_precision_mean:.4f}")
print(f"Top-{TOP_K} Recall@K: {topk_recall_mean:.4f}")
print(f"Macro F1-score: {f1_macro:.4f}")
print(f"Micro F1-score: {f1_micro:.4f}")
print(f"Micro Precision: {precision_micro:.4f}")
print(f"Micro Recall: {recall_micro:.4f}")
print("--------------------------------------------------")
