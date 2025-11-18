import os

# --- THÔNG SỐ CHUNG ---
RANDOM_STATE = 42

# --- ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Thư mục gốc ml_service
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Đường dẫn đến thư mục chứa TẤT CẢ các ảnh
# (Giả sử thư mục ảnh nằm trong data/raw/images/)
IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'images') # !!! KIỂM TRA LẠI TÊN THƯ MỤC NÀY

# --- CẤU HÌNH NHÃN ---
# Đường dẫn đến file map nhãn đã tạo ở Bước 3
LABELS_MAP_PATH = os.path.join(PROCESSED_DIR, 'labels_map.json')

# --- CẤU HÌNH MÔ HÌNH ---
# Tên các mô hình pre-trained từ Hugging Face
TEXT_MODEL_NAME = "vinai/phobert-base"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224-in21k"

# --- CẤU HÌNH XỬ LÝ DỮ LIỆU ---
IMAGE_SIZE = 224
MAX_TOKEN_LENGTH = 128 # Độ dài tối đa của câu cho PhoBERT

# --- CẤU HÌNH HUẤN LUYỆN ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15

# --- TÊN CỘT TRONG CSV ---
# (Phải khớp với file CSV của bạn)
TEXT_COLUMN = 'text'        # !!! Cột chứa nội dung text
IMAGE_COLUMN = 'image_path' # !!! Cột chứa tên file ảnh (ví dụ: timg_382.jpg)
LABEL_COLUMN = 'label'      # !!! Cột chứa nhãn (ví dụ: "thể thao, bóng chuyền")