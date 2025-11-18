import os
import pandas as pd
import re

### CẤU HÌNH ###
RAW_CSV_NAME = 'postfinal.csv'  # Tên file CSV
LABEL_COLUMN = 'label'  # Tên cột nhãn
##################

print("--- BẮT ĐẦU CHUẨN HÓA (LÀM SẠCH) NHÃN DỮ LIỆU (V3 - TÁCH NHÃN) ---")

# 1. Định nghĩa đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', RAW_CSV_NAME)
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')

files_to_fix = [
    RAW_DATA_PATH,
    os.path.join(SPLITS_DIR, 'train.csv'),
    os.path.join(SPLITS_DIR, 'val.csv'),
    os.path.join(SPLITS_DIR, 'test.csv')
]


def normalize_label_v3(label_str):
    """
    Quy trình chuẩn hóa nhãn
    - Chuẩn hóa (lower, strip)
    - Xử lý các quy tắc TÁCH và KHÔNG TÁCH đặc biệt.

    Returns:
        Một chuỗi (str). Nếu cần tách, nó sẽ trả về 1 chuỗi
        đã được nối bằng dấu phẩy (ví dụ: "chính trị,kinh tế").
    """
    if not isinstance(label_str, str):
        return ""  # Trả về rỗng nếu là NaN

    # 1. Chuẩn hóa cơ bản
    clean_label = label_str.lower().strip()

    # === 2. XỬ LÝ QUY TẮC ĐẶC BIỆT ===

    # QUY TẮC TÁCH 1: "Chính trị kinh tế"
    if 'chính trị' in clean_label and 'kinh tế' in clean_label:
        # Trả về 2 nhãn, nối bằng dấu phẩy
        return 'chính trị,kinh tế'

        # QUY TẮC TÁCH 2: "Văn hóa xã hội"
    if 'văn hóa' in clean_label and 'xã hội' in clean_label:
        return 'văn hóa,xã hội'

    #QUY TẮC TÁCH 3: "Kinh tế xã hội"
    if 'kinh tế' in clean_label and 'xã hội' in clean_label:
        return 'kinh tế,xã hội'

    # # QUY TẮC TÁCH 4: "thể thao, võ thuật"
    # if 'thể thao' in clean_label and 'võ thuật' in clean_label:
    #     return 'thể thao'
    #
    # # QUY TẮC TÁCH 5: "thể thao, bóng chuyền"
    # if 'thể thao' in clean_label and 'bóng chuyền' in clean_label:
    #     return 'thể thao'
    # #Quy tắc tách 6: "thể thao, chạy bộ"
    # if 'thể thao' in clean_label and 'chạy bộ' in clean_label:
    #     return 'thể thao'


    # QUY TẮC KHÔNG TÁCH: "Thực phẩm đồ uống"
    if 'thực phẩm' in clean_label or 'đồ uống' in clean_label:
        return 'thực phẩm đồ uống'  # Gộp tất cả về 1 tên chuẩn

    # === 3. XỬ LÝ CÁC NHÃN CÒN LẠI ===

    # Xóa bỏ các dấu nối còn lại (nếu có)
    clean_label = clean_label.replace(' - ', ' ')
    clean_label = clean_label.replace('-', ' ')
    clean_label = clean_label.replace('&', ' ')

    # Thu gọn nhiều dấu cách thành 1
    clean_label = ' '.join(clean_label.split())

    return clean_label


total_changes = 0

# 2. Lặp qua và sửa từng file
for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"\nBỏ qua: Không tìm thấy file {file_path}.")
        continue

    print(f"\nĐang xử lý file: {file_path}")

    try:
        df = pd.read_csv(file_path)

        if LABEL_COLUMN not in df.columns:
            print(f"  LỖI: Không tìm thấy cột '{LABEL_COLUMN}'. Bỏ qua.")
            continue


        # Áp dụng hàm chuẩn hóa cho từng nhãn trong cột
        def clean_multilabel_string_v3(s):
            if not isinstance(s, str):
                return ""

            original_labels = s.split(',')
            final_cleaned_labels = []

            for l in original_labels:
                # Gửi từng nhãn con đi chuẩn hóa
                cleaned_str = normalize_label_v3(l)

                # Thêm vào danh sách cuối
                # (Nếu rỗng thì bỏ qua)
                if cleaned_str:
                    final_cleaned_labels.append(cleaned_str)

            # Nối tất cả lại bằng dấu phẩy
            return ','.join(final_cleaned_labels)


        original_labels = df[LABEL_COLUMN].copy()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(clean_multilabel_string_v3)

        changes = (original_labels != df[LABEL_COLUMN]).sum()
        total_changes += changes

        if changes > 0:
            print(f"  Tìm thấy và chuẩn hóa/tách {changes} dòng.")
            df.to_csv(file_path, index=False)
            print(f"  Đã lưu file thành công.")
        else:
            print("  Không tìm thấy nhãn nào cần chuẩn hóa/tách.")

    except Exception as e:
        print(f"  Gặp lỗi bất ngờ khi xử lý file: {e}")

print("\n--- HOÀN THÀNH ---")
print(f"Tổng cộng đã chuẩn hóa/tách {total_changes} nhãn.")
print("Dữ liệu nhãn của bạn giờ đã sạch. HÃY CHẠY LẠI TOÀN BỘ PIPELINE.")