import os
import pandas as pd

### CẤU HÌNH ###
RAW_CSV_NAME = 'postfinal.csv'  # file CSV
IMAGE_COLUMN = 'image_path'  # Tên cột chứa đường dẫn ảnh
##################

print("--- BƯỚC B: BẮT ĐẦU CẬP NHẬT FILE CSV ---")

# 1. Định nghĩa đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', RAW_CSV_NAME)
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')

# Danh sách tất cả các file CSV cần sửa
files_to_fix = [
    RAW_DATA_PATH,  # Sửa cả file gốc
    os.path.join(SPLITS_DIR, 'train.csv'),
    os.path.join(SPLITS_DIR, 'val.csv'),
    os.path.join(SPLITS_DIR, 'test.csv')
]

ERROR_PATTERN = "aimg_ "
FIX_PATTERN = "aimg_"

total_fixes_all_files = 0

# 2. Lặp qua và sửa từng file
for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"\nCẢNH BÁO: Không tìm thấy file {file_path}. Bỏ qua.")
        continue

    print(f"\nĐang xử lý file: {file_path}")

    try:
        df = pd.read_csv(file_path)

        if IMAGE_COLUMN not in df.columns:
            print(f"  LỖI: Không tìm thấy cột '{IMAGE_COLUMN}'. Bỏ qua.")
            continue

        # Kiểm tra xem cột có phải kiểu string không
        if not pd.api.types.is_string_dtype(df[IMAGE_COLUMN]):
            df[IMAGE_COLUMN] = df[IMAGE_COLUMN].astype(str)

        # Tạo mask (bộ lọc) cho các dòng chứa lỗi
        mask = df[IMAGE_COLUMN].str.contains(ERROR_PATTERN, na=False)
        num_errors = mask.sum()

        if num_errors == 0:
            print("  Không tìm thấy dòng nào cần sửa.")
        else:
            print(f"  Tìm thấy {num_errors} dòng bị lỗi. Đang sửa...")

            # Sửa tất cả các dòng chứa lỗi
            df[IMAGE_COLUMN] = df[IMAGE_COLUMN].str.replace(ERROR_PATTERN, FIX_PATTERN)

            # Lưu đè file
            df.to_csv(file_path, index=False)
            print(f"  Đã sửa và lưu file thành công.")
            total_fixes_all_files += num_errors

    except Exception as e:
        print(f"  Gặp lỗi bất ngờ khi xử lý file: {e}")

print("\n--- HOÀN THÀNH BƯỚC B ---")
print(f"Tổng cộng đã sửa {total_fixes_all_files} dòng trên tất cả các file.")