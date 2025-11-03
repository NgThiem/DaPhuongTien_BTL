import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import joblib
from tqdm import tqdm


# ---------------- CONFIG ----------------
K = 256
STEP_SIZE = 8   # bước nhảy khi quét ảnh
BIN_SIZE = 16   # kich thuoc patch (patch bin_size x bin_size)


# Hàm trích chọn đặc trưng
def detectAndCompute(img, step_size=STEP_SIZE, bin_size=BIN_SIZE):
    h, w = img.shape
    descriptors = []  # danh sach luu descriptor 128 chieu
    keypoints = []    # danh sach luu vi tri keypoint (x, y)

    # Tinh gradient theo x va y
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    # Tinh do lon gradient va huong gradient
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 360

    # Lap qua anh theo cac patch voi step_size
    for y in range(bin_size//2, h-bin_size//2, step_size):
        for x in range(bin_size//2, w-bin_size//2, step_size):
            # Lay patch bin_size x bin_size quanh diem (x,y)
            patch_mag = magnitude[y-bin_size//2:y+bin_size//2, x-bin_size//2:x+bin_size//2]
            patch_angle = angle[y-bin_size//2:y+bin_size//2, x-bin_size//2:x+bin_size//2]

            descriptor = []  # descriptor 128 chieu cho patch hien tai
            sub_size = bin_size // 4  # chia patch thanh 4x4 cell

            # Lap qua tung cell 4x4 trong patch
            for i in range(4):
                for j in range(4):
                    sub_mag = patch_mag[i*sub_size:(i+1)*sub_size, j*sub_size:(j+1)*sub_size]
                    sub_angle = patch_angle[i*sub_size:(i+1)*sub_size, j*sub_size:(j+1)*sub_size]
                    hist = np.zeros(8, dtype=np.float32)  # histogram 8 huong cho cell

                    # Lap qua tat ca diem trong cell de tinh histogram
                    for m, a in zip(sub_mag.flatten(), sub_angle.flatten()):
                        bin_idx = int(a // 45) % 8  # chia 360 do thanh 8 bin
                        hist[bin_idx] += m         # cong do lon gradient vao bin tuong ung

                    descriptor.extend(hist)  # noi histogram cell vao descriptor patch

            descriptor = np.array(descriptor, dtype=np.float32)  # chuyen descriptor thanh numpy array
            if descriptor.shape[0] == 128:  # chi luu neu du 128 chieu
                descriptors.append(descriptor)
                keypoints.append((x, y))  # luu vi tri keypoint

    descriptors = np.array(descriptors, dtype=np.float32)
    return keypoints, descriptors  # tra ve keypoints va descriptors
# ---------------- Extract SIFT descriptors ----------------
def kmeans(descriptors, K=256, max_iter=20):
    idx = np.random.choice(len(descriptors), K, replace=False)
    centroids = descriptors[idx].copy()

    for it in range(max_iter):
        print(f"KMeans iter {it+1}/{max_iter}")
        assignments = []
        for d in descriptors:
            distances = np.sum((centroids - d)**2, axis=1)
            cluster_idx = np.argmin(distances)
            assignments.append(cluster_idx)
        assignments = np.array(assignments)
        for k in range(K):
            members = descriptors[assignments == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)
    return centroids

# ---------------- Tao BoVW histogram ----------------
def bovw_hist(descriptors, centroids):
    hist = np.zeros(len(centroids), dtype=np.float32)
    if len(descriptors) == 0:
        return hist
    for d in descriptors:
        distances = np.sum((centroids - d)**2, axis=1)
        idx = np.argmin(distances)
        hist[idx] += 1
    norm = np.sqrt(np.sum(hist ** 2))
    if norm > 1e-6:
        hist /= norm
    return hist

# ----------------- CHỈ CHẠY KHI FILE NÀY ĐƯỢC CHẠY TRỰC TIẾP -----------------
if __name__ == "__main__":
    IMG_DIR = Path("D:/DaPhuongTien/dataset_cleaned")
    BOVW_FILE = Path("D:/DaPhuongTien/bovw_vectors.pkl")
    KMEANS_FILE = Path("D:/DaPhuongTien/bovw_kmeans.pkl")

    image_paths = [p for p in IMG_DIR.rglob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"Tim thay {len(image_paths)} anh")

    all_descriptors = []
    image_des_list = []

    for path in tqdm(image_paths, desc="Extract descriptors"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = detectAndCompute(img)
        image_des_list.append((path, des))
        if des.shape[0] > 0:
            all_descriptors.append(des)
    all_descriptors = np.vstack(all_descriptors)
    print(f"Tong so descriptor: {all_descriptors.shape}")

    kmeans_model = MiniBatchKMeans(n_clusters=K, max_iter=300, random_state=42)
    kmeans_model.fit(all_descriptors)
    joblib.dump(kmeans_model, KMEANS_FILE)
    print(f"Da luu KMeans centroids vao {KMEANS_FILE}")
    centroids = kmeans_model.cluster_centers_

    bovw_vectors = {}
    for path, des in tqdm(image_des_list, desc="Creating BoVW vectors"):
        hist = bovw_hist(des, centroids)
        key = str(path).replace('\\','/')
        bovw_vectors[key] = hist
    joblib.dump(bovw_vectors, BOVW_FILE)
    print(f"Da luu BoVW vectors vao {BOVW_FILE}")