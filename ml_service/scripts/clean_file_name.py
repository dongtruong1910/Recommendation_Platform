import os

print("--- BƯỚC A: BẮT ĐẦU DỌN DẸP TÊN FILE ẢNH ---")

# 1. Định nghĩa đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'images')

if not os.path.isdir(IMAGE_DIR):
    print(f"LỖI: Không tìm thấy thư mục ảnh tại: {IMAGE_DIR}")
    exit()

print(f"Đang quét thư mục: {IMAGE_DIR}")

# Các file 'aimg_' có lỗi sẽ có dạng "aimg_ " (có 1 dấu cách)
ERROR_PATTERN = "aimg_ "
FIX_PATTERN = "aimg_"  # Sửa thành không có dấu cách

total_files_scanned = 0
total_files_renamed = 0

# 2. Quét và đổi tên
for filename in os.listdir(IMAGE_DIR):
    total_files_scanned += 1

    # Kiểm tra xem file có chứa lỗi không
    if ERROR_PATTERN in filename:
        try:
            # Tạo tên mới và đường dẫn cũ/mới
            new_filename = filename.replace(ERROR_PATTERN, FIX_PATTERN)
            old_file_path = os.path.join(IMAGE_DIR, filename)
            new_file_path = os.path.join(IMAGE_DIR, new_filename)

            # Thực hiện đổi tên
            os.rename(old_file_path, new_file_path)

            print(f"  ĐÃ ĐỔI TÊN: {filename}  --->  {new_filename}")
            total_files_renamed += 1

        except Exception as e:
            print(f"  LỖI khi đổi tên {filename}: {e}")

print("\n--- HOÀN THÀNH BƯỚC A ---")
print(f"Đã quét {total_files_scanned} file.")
print(f"Đã đổi tên thành công {total_files_renamed} file.")