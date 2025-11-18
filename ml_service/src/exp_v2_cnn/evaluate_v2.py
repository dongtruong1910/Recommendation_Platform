import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd


try:
    from src import config
    from src.exp_v2_cnn.dataset_v2 import create_data_loader_v2
    from src.exp_v2_cnn.model_v2 import MultimodalClassifierV2  # Import Class
except ImportError:
    # Thử import theo cách "module" (nếu chạy bằng -m)
    from .. import config
    from .dataset_v2 import create_data_loader_v2
    from .model_v2 import MultimodalClassifierV2


def plot_evaluation_report(report_dict, save_path, model_version="V1"):
    """
    Hàm này nhận report (dict) và vẽ BIỂU ĐỒ CỘT NHÓM (P, R, F1).
    """
    print(f"Đang vẽ biểu đồ chi tiết và lưu tại: {save_path}")
    try:
        # Chuyển dict sang DataFrame của Pandas
        df = pd.DataFrame(report_dict).transpose()

        plot_df = df.drop(index=['macro avg', 'weighted avg', 'micro avg', 'samples avg'], errors='ignore')
        # ==========================

        #lấy 3 cột
        plot_df = plot_df[['precision', 'recall', 'f1-score']]

        # Sắp xếp theo F1-Score
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

def run_evaluation_v2(model_path, threshold):
    """
    Hàm chính để chạy đánh giá V2 trên tập TEST.
    """

    print("--- BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH V2 (CustomCNN) TRÊN TẬP TEST ---")

    # 1. Thiết lập Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải Dữ liệu Test
    print("Đang tải Test DataLoader V2 (Otsu/Morph)...")
    try:
        test_loader, test_dataset = create_data_loader_v2('test')
        num_classes = test_dataset.num_classes
        labels_map = test_dataset.labels_map
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu V2: {e}")
        return

    print(f"Đã tải {len(test_dataset)} mẫu test.")

    # 3. Khởi tạo và Tải Mô hình V2
    print(f"Đang tải mô hình V2 từ: {model_path}")

    # Khởi tạo kiến trúc V2 (PhoBERT + CustomCNN với Sobel)
    model = MultimodalClassifierV2(num_classes=num_classes, use_sobel_layer=True)

    try:
        # Tải trọng số đã train
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file mô hình {model_path}")
        print("Bạn đã chạy 'train_v2.py' chưa?")
        return
    except Exception as e:
        print(f"LỖI khi tải model state_dict: {e}")
        return

    model.to(device)
    model.eval()

    # 4. Chạy Vòng lặp Dự đoán
    print("Đang chạy dự đoán V2 trên tập test...")
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set V2"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor = batch['image_tensor'].to(device)  # Ảnh 1 kênh
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, image_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).int()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    print("Đã hoàn thành dự đoán V2.")

    # 5. Tính toán và In Kết quả
    print("\n--- KẾT QUẢ ĐÁNH GIÁ V2 (CustomCNN + IP) ---")
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

    print("\n--- BÁO CÁO CHI TIẾT TỪNG NHÃN (V2) ---")
    target_names = [label for label, idx in sorted(labels_map.items(), key=lambda item: item[1])]

    try:
        # Lấy report dưới dạng STRING để in ra
        report_str = classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0)
        print(report_str)

        # Lấy report dưới dạng DICT để vẽ biểu đồ
        report_dict = classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0,
                                            output_dict=True)

        # Định nghĩa đường dẫn lưu
        save_path = os.path.join(config.BASE_DIR, 'models', 'evaluation_report_v2_cnn.png')

        # Gọi hàm vẽ
        plot_evaluation_report(report_dict, save_path, model_version="V2 (CustomCNN + IP)")

    except Exception as e:
        print(f"Lỗi khi tạo/vẽ classification report: {e}")

    print("--- HOÀN THÀNH ĐÁNH GIÁ V2 ---")


# Chạy hàm chính
if __name__ == "__main__":
    # --- CẤU HÌNH ---
    # 1. Tên file model V2
    MODEL_NAME = 'best_model_v2_cnn.pth'

    # 2. Ngưỡng quyết định
    PREDICTION_THRESHOLD = 0.5
    # --------------

    model_path = os.path.join(config.BASE_DIR, 'models', MODEL_NAME)

    run_evaluation_v2(
        model_path=model_path,
        threshold=PREDICTION_THRESHOLD
    )