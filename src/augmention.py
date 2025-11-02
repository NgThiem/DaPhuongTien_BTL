import os
from pathlib import Path
import cv2
import numpy as np
import random
import shutil
import csv

# ==== Cấu hình ====
RAW_DIR = Path("data/raw")        # 2 folder: flower / fruit
AUG_DIR = Path("data/augmented")  # Lưu ảnh gốc + tăng cường theo lớp
SPLIT_DIR = Path("data/split")    # Lưu train/val/test theo lớp
RANDOM_SEED = 42
AUG_PER_IMAGE = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ==== Hàm tăng cường ảnh ====
def augment_image(img):
    h, w = img.shape[:2]
    imgs = []

    # 1. Lật ngang
    imgs.append(cv2.flip(img, 1))
    # 2. Xoay ±15°
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    imgs.append(cv2.warpAffine(img, M, (w, h)))
    # 3. Thay đổi độ sáng
    alpha = np.random.uniform(0.7, 1.3)
    beta = np.random.randint(-30, 30)
    imgs.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))
    # 4. Crop nhỏ
    crop_ratio = 0.9
    ch, cw = int(h*crop_ratio), int(w*crop_ratio)
    start_h = np.random.randint(0, h - ch + 1)
    start_w = np.random.randint(0, w - cw + 1)
    crop = img[start_h:start_h+ch, start_w:start_w+cw]
    crop = cv2.resize(crop, (w, h))
    imgs.append(crop)

    return imgs

# ==== Tăng cường dữ liệu theo lớp ====
for main_dir in RAW_DIR.iterdir():  # flower / fruit
    if not main_dir.is_dir():
        continue

    for cls_dir in main_dir.iterdir():  # rose, apple, ...
        if not cls_dir.is_dir():
            continue
        class_name = cls_dir.name
        out_class_dir = AUG_DIR / class_name
        os.makedirs(out_class_dir, exist_ok=True)
        print(f"Tang cuong class: {class_name}")

        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print("Khong doc duoc:", img_path)
                continue

            # Lưu ảnh gốc
            cv2.imwrite(str(out_class_dir / img_path.name), img)

            # Tăng cường
            aug_imgs = augment_image(img)
            for i, aug in enumerate(aug_imgs[:AUG_PER_IMAGE]):
                new_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(out_class_dir / new_name), aug)

# ==== Chia train/val/test theo lớp ====
metadata = []

for class_dir in AUG_DIR.iterdir():  # mỗi lớp
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    imgs = [f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train+n_val],
        "test": imgs[n_train+n_val:]
    }

    for split_name, split_imgs in splits.items():
        split_cls_dir = SPLIT_DIR / split_name / class_name
        os.makedirs(split_cls_dir, exist_ok=True)
        for img_path in split_imgs:
            dst = split_cls_dir / img_path.name
            shutil.copy(img_path, dst)
            metadata.append([dst, class_name, split_name])

# ==== Ghi metadata CSV ====
csv_file = SPLIT_DIR / "metadata.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "split"])
    writer.writerows(metadata)

print("Done.")
