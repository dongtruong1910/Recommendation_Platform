import torch
import torch.nn as nn
from transformers import AutoTokenizer
from PIL import Image
import json
import os
import numpy as np
from torchvision import transforms
from typing import List

# --- Import các module V2 ---
try:
    from src import config

    # 1. Import TÊN CLASS model V2
    from src.exp_v2_cnn.model_v2 import MultimodalClassifierV2

    # 2. Import các hàm XỬ LÝ ẢNH từ ip_utils.py
    from src.exp_v2_cnn.ip_utils import otsu_threshold, morph_opening, morph_closing

except ImportError as e:
    print(f"LỖI: Không thể import V2 modules: {e}")
    print("Hãy đảm bảo file 'ip_utils.py' nằm trong 'src/exp_v2_cnn/'")
    exit()


class PredictorV2:
    """
    Class này bao bọc logic dự đoán cho model V2 (PhoBERT + CustomCNN).
    """

    def __init__(self, model_path):
        print("--- Đang khởi tạo Predictor V2 (CNN) ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

        # 1. Tải bản đồ nhãn
        try:
            with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
                self.labels_map = json.load(f)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file {config.LABELS_MAP_PATH}")
            exit()

        self.idx_to_label = {idx: label for label, idx in self.labels_map.items()}
        self.num_classes = len(self.labels_map)
        print(f"Đã tải {self.num_classes} nhãn.")

        # 2. Tải Tokenizer (cho text)
        print(f"Đang tải Tokenizer: {config.TEXT_MODEL_NAME}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # 3. Tải Image Transform V2 (SAO CHÉP TỪ DATASET_V2.PY)
        self.image_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(),  # Chuyển sang 1 kênh
            np.array,  # Chuyển sang numpy
            transforms.Lambda(self.apply_ip_pipeline),  # Áp dụng Otsu, Morph
            transforms.ToTensor(),  # Chuyển sang Tensor
        ])
        print("Đã tạo pipeline xử lý ảnh V2 (Otsu + Morph).")

        # 4. Tải Mô hình V2
        print(f"Đang tải mô hình V2 từ: {model_path}")
        self.model = MultimodalClassifierV2(num_classes=self.num_classes, use_sobel_layer=True)

        # Tải trọng số
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file mô hình {model_path}")
            exit()

        self.model.to(self.device)
        self.model.eval()

        print("--- Predictor V2 đã sẵn sàng! ---")

    # ---  ---
    def apply_ip_pipeline(self, image_gray_numpy):
        """Sao chép y hệt logic từ dataset."""
        threshold = otsu_threshold(image_gray_numpy)
        binary_image = image_gray_numpy > threshold
        cleaned_image = morph_opening(binary_image, kernel_size=3)
        cleaned_image = morph_closing(cleaned_image, kernel_size=3)
        return cleaned_image.astype(np.uint8) * 255

    # ---  ---

    def _process_text(self, text_content):
        """Mã hóa text đầu vào."""
        text_inputs = self.text_tokenizer(
            text_content,
            max_length=config.MAX_TOKEN_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return text_inputs['input_ids'].to(self.device), text_inputs['attention_mask'].to(self.device)

    def _process_image(self, image_path):
        """Tải và xử lý ảnh đầu vào (cho V2)."""
        try:
            with open(image_path, 'rb') as f:
                image_pil = Image.open(f)
                # Sao chép y hệt logic của __getitem__: convert('RGB')
                image = image_pil.convert('RGB')
        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy ảnh {image_path}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')
        except Exception as e:
            print(f"LỖI khi đọc ảnh {image_path}: {e}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')

        # Áp dụng transform V2 và thêm 1 chiều "batch"
        # Transform này sẽ tự chuyển nó sang Grayscale, Otsu, v.v.
        return self.image_transform(image).unsqueeze(0).to(self.device)

    #
    # --- HÀM PREDICT ---
    #
    def predict(self, text_content: str, image_path: str, threshold=0.5):
        """
        Dự đoán nhãn cho (1 text, 1 ảnh) dùng model V2.
        """

        # 1. Xử lý đầu vào
        input_ids, attention_mask = self._process_text(text_content)
        image_tensor = self._process_image(image_path)  # Xử lý 1 ảnh

        # 2. Chạy dự đoán
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, image_tensor)

        # 3. Tính xác suất
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.cpu().numpy()[0]

        # 4. Quyết định nhãn
        results = {
            "predicted_labels": [],
            "all_probabilities": {}
        }
        for i in range(self.num_classes):
            label_name = self.idx_to_label[i]
            prob = probabilities[i]

            results["all_probabilities"][label_name] = float(prob)

            if prob >= threshold:
                results["predicted_labels"].append(label_name)

        return results


if __name__ == "__main__":
    import sys

    # 1. Lấy thư mục gốc (ml_service)
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
    if os.path.basename(SRC_DIR) == 'exp_v2_cnn':
        SRC_DIR = os.path.dirname(SRC_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # 2. Thêm thư mục gốc vào path để import src
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
        print(f"Đã thêm {ROOT_DIR} vào sys.path")


    # --- CẤU HÌNH TEST ---
    TEST_IMAGE_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'images', 'img.png')
    TEST_TEXT = "Messi vừa ghi bàn thắng ấn định tỉ số của trận đấu"

    MODEL_V2_PATH = os.path.join(ROOT_DIR, 'models', 'best_model_v2_cnn.pth')
    THRESHOLD = 0.5
    # --- ---

    print("\n--- KIỂM TRA PredictorV2 ---")

    # 1. Kiểm tra file
    if not os.path.exists(MODEL_V2_PATH):
        print(f"LỖI: Không tìm thấy model V2 tại: {MODEL_V2_PATH}")
        exit()
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"CẢNH BÁO: Không tìm thấy ảnh test: {TEST_IMAGE_PATH}. Dùng ảnh đen.")
        # Vẫn tiếp tục, predictor sẽ dùng ảnh đen

    # 2. Khởi tạo
    try:
        predictor = PredictorV2(model_path=MODEL_V2_PATH)
    except Exception as e:
        print(f"Lỗi khi khởi tạo PredictorV2: {e}")
        import traceback

        traceback.print_exc()
        exit()

    # 3. Dự đoán
    print("\n--- Đang dự đoán V2 ---")
    results = predictor.predict(
        text_content=TEST_TEXT,
        image_path=TEST_IMAGE_PATH,
        threshold=THRESHOLD
    )

    # 4. In kết quả
    print("\n--- KẾT QUẢ DỰ ĐOÁN V2 ---")
    print(f"(Với ngưỡng = {THRESHOLD})")
    print(f"\nCác nhãn được dự đoán:")
    if results["predicted_labels"]:
        for label in results["predicted_labels"]:
            print(f"  - {label} (Score: {results['all_probabilities'][label]:.4f})")
    else:
        print("  (Không có nhãn nào vượt ngưỡng)")

    print("\n--- (Chi tiết 5 score cao nhất) ---")
    sorted_probs = sorted(results["all_probabilities"].items(), key=lambda item: item[1], reverse=True)
    for label, prob in sorted_probs[:5]:
        print(f"  {label}: {prob:.4f}")