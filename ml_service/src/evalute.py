import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Import các file V1 (theo cách của PyCharm)
try:
    from src import config
    from src.dataset import create_data_loader
    from src.model import MultimodalClassifier  # Import Class
except ImportError:
    # Thử import theo cách "module" (nếu chạy bằng -m)
    from .. import config
    from .dataset import create_data_loader
    from .model import MultimodalClassifier


def plot_evaluation_report(report_dict, save_path, model_version="V1 (Pre-trained)"):
    """
    Hàm này nhận report (dict) và vẽ BIỂU ĐỒ CỘT NHÓM (P, R, F1).
    (Phiên bản đã sửa lỗi KeyError 'accuracy')
    """
    print(f"Đang vẽ biểu đồ chi tiết và lưu tại: {save_path}")
    try:
        # Chuyển dict sang DataFrame của Pandas
        df = pd.DataFrame(report_dict).transpose()

        # Bỏ 'accuracy' (vì nó không có) và các dòng 'avg' khác
        plot_df = df.drop(index=['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg'], errors='ignore')

        # Chỉ lấy 3 cột chúng ta quan tâm
        plot_df = plot_df[['precision', 'recall', 'f1-score']]

        # Sắp xếp theo F1-Score để biểu đồ đẹp hơn
        plot_df = plot_df.sort_values(by='f1-score', ascending=True)  # Sắp xếp tăng dần

        # Vẽ biểu đồ cột nhóm NGANG
        ax = plot_df.plot(
            kind='barh',
            figsize=(15, 8 + len(plot_df) * 0.3),  # Kích thước động theo số nhãn
            width=0.8  # Độ rộng của nhóm
        )

        plt.title(f"Báo cáo chi tiết Precision, Recall, F1 (Mô hình {model_version})")
        plt.xlabel("Điểm số (Score)")
        plt.ylabel("Nhãn (Labels)")
        plt.legend(loc='lower right')
        plt.tight_layout()  # Tự động căn chỉnh
        plt.savefig(save_path)  # Lưu file
        print(f"Đã lưu biểu đồ {model_version} tại: {save_path}")

    except Exception as e:
        print(f"LỖI khi vẽ biểu đồ: {e}")
        import traceback
        traceback.print_exc()
        print("Lưu ý: Bạn có thể cần cài đặt font tiếng Việt cho matplotlib nếu tên nhãn bị lỗi.")


def run_evaluation(model_path, is_finetuned, threshold):
    """
    Hàm chính để chạy đánh giá V1 trên tập TEST.
    """

    print("--- BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH V1 (Pre-trained) TRÊN TẬP TEST ---")

    # 1. Thiết lập Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải Dữ liệu Test (Dùng loader V1)
    print("Đang tải Test DataLoader V1 (RGB)...")
    try:
        test_loader, test_dataset = create_data_loader('test')
        num_classes = test_dataset.num_classes
        labels_map = test_dataset.labels_map
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu V1: {e}")
        return

    print(f"Đã tải {len(test_dataset)} mẫu test.")

    # 3. Khởi tạo và Tải Mô hình V1
    print(f"Đang tải mô hình V1 từ: {model_path}")

    # freeze_backbones phải khớp với cách mô hình được train
    # is_finetuned=False -> Giai đoạn 1 (đóng băng)
    # is_finetuned=True  -> Giai đoạn 2 (đã mở băng)
    model = MultimodalClassifier(num_classes=num_classes, freeze_backbones=(not is_finetuned))

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file mô hình {model_path}")
        return
    except Exception as e:
        print(f"LỖI khi tải model state_dict: {e}")
        print("Kiểm tra xem bạn đã đặt cờ 'IS_FINETUNED' cho đúng chưa.")
        return

    model.to(device)
    model.eval()

    # 4. Chạy Vòng lặp Dự đoán
    print("Đang chạy dự đoán V1 trên tập test...")
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set V1"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor = batch['image_tensor'].to(device)  # Ảnh 3 kênh
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, image_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).int()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    print("Đã hoàn thành dự đoán V1.")

    # 5. Tính toán và In Kết quả
    print("\n--- KẾT QUẢ ĐÁNH GIÁ V1 (Pre-trained) ---")
    print(f"Ngưỡng dự đoán (Threshold): {threshold}")

    exact_match_ratio = accuracy_score(all_labels, all_predictions)
    print(f"\nExact Match Ratio (Accuracy): {exact_match_ratio * 100:.2f}%")

    print("\n--- Các chỉ số trung bình (Weighted) ---")
    precision_w = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall_w = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1_w = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    print(f"  Precision (Weighted): {precision_w:.4f}")
    print(f"  Recall (Weighted):    {recall_w:.4f}")
    print(f"  F1-Score (Weighted):  {f1_w:.4f}")

    print("\n--- BÁO CÁO CHI TIẾT TỪNG NHÃN (V1) ---")
    target_names = [label for label, idx in sorted(labels_map.items(), key=lambda item: item[1])]

    try:
        # Lấy report dưới dạng STRING để in ra
        report_str = classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0)
        print(report_str)

        # Lấy report dưới dạng DICT để vẽ biểu đồ
        report_dict = classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0,
                                            output_dict=True)

        # Định nghĩa đường dẫn lưu
        save_path = os.path.join(config.BASE_DIR, 'models', 'evaluation_report_v1.png')

        # Gọi hàm vẽ
        plot_evaluation_report(report_dict, save_path, model_version="V1 (Pre-trained)")

    except Exception as e:
        print(f"Lỗi khi tạo/vẽ classification report: {e}")

    print("--- HOÀN THÀNH ĐÁNH GIÁ V1 ---")


# Chạy hàm chính
if __name__ == "__main__":
    # --- CẤU HÌNH ---
    # 1. Chọn file model bạn muốn test
    MODEL_NAME = 'best_model.pth'  # (Mô hình Giai đoạn 1)
    # MODEL_NAME = 'best_finetuned_model.pth' # (Mô hình Giai đoạn 2)

    # 2. Đặt cho đúng với model ở trên
    # (Nếu dùng 'best_model.pth' -> False. Nếu dùng 'best_finetuned_model.pth' -> True)
    IS_FINETUNED = False

    # 3. Ngưỡng quyết định
    PREDICTION_THRESHOLD = 0.5
    # --------------

    model_path = os.path.join(config.BASE_DIR, 'models', MODEL_NAME)

    run_evaluation(
        model_path=model_path,
        is_finetuned=IS_FINETUNED,
        threshold=PREDICTION_THRESHOLD
    )