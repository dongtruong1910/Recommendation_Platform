import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import json

### CẤU HÌNH ###
RAW_CSV_NAME = 'postfinal.csv'  # !!! TÊN FILE CSV
LABEL_COLUMN = 'label'  # !!! TÊN CỘT NHÃN

TRAIN_SIZE = 0.7  # 70% cho tập train
VAL_SIZE = 0.15  # 15% cho tập validation
TEST_SIZE = 0.15  # 15% cho tập test
RANDOM_STATE = 42  # Để đảm bảo kết quả chia nhất quán
##################

# 1. Định nghĩa đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')  # Thư mục để lưu bản đồ nhãn

# Tạo các thư mục
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f"Đang đọc dữ liệu từ: {os.path.join(RAW_DATA_DIR, RAW_CSV_NAME)}")

# 2. Đọc file CSV gốc
try:
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_CSV_NAME))
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {RAW_CSV_NAME} trong thư mục {RAW_DATA_DIR}")
    exit()
except Exception as e:
    print(f"LỖI khi đọc file CSV: {e}")
    exit()

print(f"Đọc thành công {len(df)} dòng dữ liệu.")
print("-" * 30)

# 3. XỬ LÝ NHÃN (PHẦN MỚI CHO MULTI-LABEL)
print(f"Bắt đầu xử lý cột nhãn '{LABEL_COLUMN}'...")

if LABEL_COLUMN not in df.columns:
    print(f"LỖI: Không tìm thấy cột nhãn '{LABEL_COLUMN}'. Các cột hiện có: {list(df.columns)}")
    exit()

# Xử lý các giá trị NaN (nếu có) trong cột nhãn
df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna('').astype(str)

all_labels_set = set()


def get_labels_from_string(label_str):
    """
    Tách chuỗi "label1, label2, label3" thành list ['label1', 'label2', 'label3']
    Tách chuỗi "label1" thành list ['label1']
    """
    if not label_str.strip():
        return []
    # Tách bằng dấu phẩy, sau đó lột bỏ khoảng trắng thừa ở 2 đầu
    labels = [label.strip() for label in label_str.split(',')]
    # Lọc bỏ các nhãn rỗng (ví dụ nếu chuỗi là "a,,b")
    labels = [label for label in labels if label]
    return labels


# --- tìm tất cả nhãn duy nhất ---
for label_str in df[LABEL_COLUMN]:
    labels = get_labels_from_string(label_str)
    all_labels_set.update(labels)  # Thêm tất cả nhãn tìm được vào set

# Chuyển set thành list và sắp xếp
all_labels_list = sorted(list(all_labels_set))

# --- Tạo bản đồ (mapping) từ nhãn sang ID ---
labels_map = {label: i for i, label in enumerate(all_labels_list)}
num_classes = len(all_labels_list)

print(f"Tìm thấy {num_classes} nhãn duy nhất.")

# --- Lưu bản đồ  ---
map_path = os.path.join(PROCESSED_DIR, 'labels_map.json')
with open(map_path, 'w', encoding='utf-8') as f:
    json.dump(labels_map, f, ensure_ascii=False, indent=4)

print(f"Đã lưu bản đồ nhãn (labels_map.json) vào: {PROCESSED_DIR}")
print("-" * 30)


# 4. TẠO CỘT 'main_label' ĐỂ CHIA DỮ LIỆU
# Lấy nhãn đầu tiên làm "nhãn chính"
def get_main_label(label_str):
    labels = get_labels_from_string(label_str)
    if labels:
        return labels[0]  # Lấy nhãn đầu tiên
    return "__EMPTY__"  # Một nhãn tạm nếu dòng đó không có nhãn nào


df['main_label'] = df[LABEL_COLUMN].apply(get_main_label)

# Kiểm tra xem có bao nhiêu dòng không có nhãn
empty_labels_count = (df['main_label'] == '__EMPTY__').sum()
if empty_labels_count > 0:
    print(f"CẢNH BÁO: Có {empty_labels_count} dòng không có nhãn nào. Chúng sẽ được gom vào nhóm '__EMPTY__'.")

print("Bắt đầu chia dữ liệu (stratified) theo 'main_label'...")

# 5. PHÂN CHIA DỮ LIỆU
# Đảm bảo tổng tỷ lệ là 1
if not (TRAIN_SIZE + VAL_SIZE + TEST_SIZE) == 1.0:
    print("LỖI: Tổng tỷ lệ (TRAIN + VAL + TEST) phải bằng 1.0")
    exit()

# Tách tập Train (70%) ra khỏi (Val + Test) (30%)
split_train_temp = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - TRAIN_SIZE), random_state=RANDOM_STATE)

for train_index, temp_index in split_train_temp.split(df, df['main_label']):
    train_set = df.loc[train_index]
    temp_set = df.loc[temp_index]

print(f"Đã chia tập Train: {len(train_set)} mẫu")

# Tách tập Val (15%) và Test (15%) từ tập temp (30%)
val_test_ratio = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
split_val_test = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=RANDOM_STATE)

for val_index, test_index in split_val_test.split(temp_set, temp_set['main_label']):
    val_set = temp_set.iloc[val_index]
    test_set = temp_set.iloc[test_index]

print(f"Đã chia tập Validation: {len(val_set)} mẫu")
print(f"Đã chia tập Test: {len(test_set)} mẫu")
print("-" * 30)

# 6. LƯU KẾT QUẢ
train_set = train_set.drop(columns=['main_label'])
val_set = val_set.drop(columns=['main_label'])
test_set = test_set.drop(columns=['main_label'])

train_path = os.path.join(SPLITS_DIR, 'train.csv')
val_path = os.path.join(SPLITS_DIR, 'val.csv')
test_path = os.path.join(SPLITS_DIR, 'test.csv')

train_set.to_csv(train_path, index=False)
val_set.to_csv(val_path, index=False)
test_set.to_csv(test_path, index=False)

print(f"Đã lưu 3 files vào thư mục: {SPLITS_DIR}")
print(f" - {os.path.basename(train_path)} ({len(train_set)} dòng)")
print(f" - {os.path.basename(val_path)} ({len(val_set)} dòng)")
print(f" - {os.path.basename(test_path)} ({len(test_set)} dòng)")
print("\nHoàn thành Bước 3 (phiên bản Multi-label)!")