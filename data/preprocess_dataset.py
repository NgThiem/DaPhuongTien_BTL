import os
from pathlib import Path #giúp thao tác đường dẫn dễ hơn, thay cho việc nối chuỗi thủ công "folder/" + "file.jpg".
from PIL import Image #thư viện Pillow, chuyên để đọc, chuyển đổi định dạng và xử lý ảnh
from tqdm import tqdm #tạo thanh tiến trình hiển thị % xử lý

# ---------------- CONFIG ----------------
RAW_DATASET_DIR = Path("data/split/train")            # Gốc dữ liệu gốc 
OUTPUT_DIR = Path("dataset_cleaned")         # Nơi lưu ảnh đã xử lý
IMAGE_SIZE = (224, 224)                      # Kích thước chuẩn cho CNN
VALID_EXT = [".jpg", ".jpeg", ".png"]        # Định dạng hợp lệ

# ---------------- TẠO THƯ MỤC OUTPUT ----------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#mkdir(parents=True): tạo thư mục dataset_cleaned/, và nếu chưa có các thư mục cha thì cũng tạo luôn.
#exist_ok=True: không báo lỗi nếu thư mục này đã tồn tại. Giúp chương trình an toàn khi chạy lại nhiều lần.

# ---------------- HÀM TIỆN ÍCH ----------------
def is_valid_image(path: Path):
    """Kiem tra anh co duoc mo khong"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
'''
Giải thích chi tiết:
Image.open(path): thử mở ảnh.
img.verify(): kiểm tra tính toàn vẹn file, không tải toàn bộ dữ liệu ảnh vào RAM, chỉ xác nhận file có bị lỗi header, corrupt... không.
Nếu ảnh hợp lệ → trả True, nếu lỗi (ảnh hỏng, không đọc được) → bắt Exception và trả False.
Tác dụng:
Giúp bỏ qua các file ảnh lỗi (ví dụ: ảnh bị hỏng khi download, kích thước 0 byte, hoặc không phải ảnh thật)
'''
def process_and_save_image(input_path: Path, output_path: Path):
    """Resize, convert RGB va luu lai"""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
    except Exception as e:
        print(f"Loi xu ly {input_path}: {e}")
'''
Giải thích chi tiết:
Image.open(input_path): mở ảnh gốc.
img.convert("RGB"): chuyển ảnh về 3 kênh màu (RGB) — rất cần thiết vì:
Một số ảnh có thể là định dạng RGBA (có alpha) hoặc L (grayscale).
CNN yêu cầu đầu vào là 3 kênh (R,G,B), nếu không sẽ lỗi.
img.resize(IMAGE_SIZE): resize về (224,224).
output_path.parent.mkdir(...): tạo sẵn thư mục con tương ứng (ví dụ dataset_cleaned/hoa_hong/).
img.save(output_path): lưu ảnh sau khi xử lý.
try/except: tránh crash toàn bộ chương trình nếu một ảnh bị lỗi.
'''
# ---------------- MAIN ----------------
if __name__ == "__main__":
    print(f"Dang quet du lieu: {RAW_DATASET_DIR}")
    all_images = list(RAW_DATASET_DIR.rglob("*")) #quét toàn bộ file trong thư mục gốc (và thư mục con).
    image_paths = [p for p in all_images if p.suffix.lower() in VALID_EXT] #Lọc ra những file có đuôi hợp lệ (VALID_EXT)
 
    print(f"Tim thay {len(image_paths)} anh hop le.") 
    for path in tqdm(image_paths, desc="Dang xu ly anh"): #tqdm(image_paths, desc=...): hiển thị tiến trình (vd: 432/2400 ảnh).
        if is_valid_image(path): # kiểm tra ảnh có đọc được không.
            # Giữ cấu trúc thư mục con 
            rel_path = path.relative_to(RAW_DATASET_DIR)
            output_path = OUTPUT_DIR / rel_path #thực hiện resize và lưu.
            process_and_save_image(path, output_path)
        else:
            print(f"Anh loi hoac khong doc duoc: {path}") #nếu ảnh hỏng → in ra để biết ảnh nào lỗi.
    print("="*60)
    print("Hoan tat xu ly!")
    print(f"Anh da duoc luu tai: {OUTPUT_DIR}")
    print("="*60)
