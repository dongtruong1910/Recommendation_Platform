import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import os
import json
from typing import List # Thêm import này

# Import các file khác của chúng ta
try:
    from src import config
    from src.model import MultimodalClassifier  # Import class mô hình
    from src.dataset import get_image_transforms  # Lấy lại hàm transform ảnh
except ImportError:
    print("Lỗi: Không thể import config, model, hoặc dataset.")
    print("Hãy đảm bảo bạn đang chạy file này từ thư mục gốc `ml_service`.")
    exit()


class Predictor:
    """
    Class này bao bọc toàn bộ logic dự đoán:
    1. Tải mô hình và các thành phần 1 LẦN DUY NHẤT.
    2. Cung cấp hàm `predict()` để dùng nhiều lần.
    """

    def __init__(self, model_path):
        print("--- Đang khởi tạo Predictor ---")

        # 1. Thiết lập Device (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

        # 2. Tải bản đồ nhãn (Labels Map)
        try:
            with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
                self.labels_map = json.load(f)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file {config.LABELS_MAP_PATH}")
            exit()

        # Tạo bản đồ ngược (ID -> Tên nhãn)
        self.idx_to_label = {idx: label for label, idx in self.labels_map.items()}
        self.num_classes = len(self.labels_map)
        print(f"Đã tải {self.num_classes} nhãn.")

        # 3. Tải Tokenizer (cho text)
        print(f"Đang tải Tokenizer: {config.TEXT_MODEL_NAME}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # 4. Tải Image Transform (cho ảnh)
        self.image_transform = get_image_transforms()

        # 5. Tải Mô hình
        print(f"Đang tải mô hình từ: {model_path}")
        self.model = MultimodalClassifier(num_classes=self.num_classes, freeze_backbones=True)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=True))
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file mô hình {model_path}")
            exit()

        # Chuyển mô hình lên GPU và đặt ở chế độ eval()
        self.model.to(self.device)
        self.model.eval()

        print("--- Predictor đã sẵn sàng! ---")

    def _process_text(self, text_content):
        """Mã hóa text đầu vào."""
        text_inputs = self.text_tokenizer(
            text_content,
            max_length=config.MAX_TOKEN_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Chuyển lên GPU ngay lập tức
        return text_inputs['input_ids'].to(self.device), text_inputs['attention_mask'].to(self.device)

    def _process_image(self, image_path):
        """Tải và xử lý ảnh đầu vào."""
        try:
            with open(image_path, 'rb') as f:
                image_pil = Image.open(f)
                image = image_pil.convert('RGB')
        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy ảnh {image_path}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')
        except Exception as e:
            print(f"LỖI khi đọc ảnh {image_path}: {e}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')

        # Áp dụng transform, thêm 1 chiều "batch" (unsqueeze) và chuyển lên GPU
        return self.image_transform(image).unsqueeze(0).to(self.device)

    #
    # --- HÀM PREDICT ĐÃ ĐƯỢC THAY ĐỔI ---
    #
    def predict(self, text_content: str, image_paths: List[str], threshold=0.5):
        """
        Dự đoán nhãn cho một cặp (text, [list_of_images]).

        Args:
            text_content (str): Nội dung văn bản của bài đăng.
            image_paths (list): DANH SÁCH các đường dẫn đến file ảnh.
            threshold (float): Ngưỡng (từ 0 đến 1) để quyết định 1 nhãn là 'True'.

        Returns:
            dict: Một dict chứa các nhãn được dự đoán và xác suất của chúng.
        """

        # 1. Xử lý text (1 lần)
        input_ids, attention_mask = self._process_text(text_content)

        # 2. Xử lý ảnh (N lần)
        image_embeddings_list = []  # List để lưu các vector đặc trưng của ảnh

        # Xử lý trường hợp không có ảnh (gửi list rỗng)
        if not image_paths:
            print("CẢNH BÁO: Không có ảnh nào được cung cấp. Sử dụng ảnh rỗng.")
            # _process_image sẽ tự tạo ảnh đen nếu đường dẫn không tồn tại
            image_paths = ["dummy_path_để_tạo_ảnh_đen"]

        # 3. Chạy dự đoán (trong 1 context no_grad)
        with torch.no_grad():
            # A. Lấy đặc trưng Text (CHỈ PHOBERT)
            # Dùng .pooler_output (như trong model.py)
            text_features = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output  # Shape: [1, 768]

            # B. Lấy đặc trưng Ảnh (CHỈ VIT) - lặp N lần
            for img_path in image_paths:
                # Tải và xử lý từng ảnh
                image_tensor = self._process_image(img_path)

                # Đưa qua mô hình ViT để lấy đặc trưng
                # Dùng .pooler_output (như trong model.py)
                img_feat = self.model.image_model(
                    pixel_values=image_tensor
                ).pooler_output  # Shape: [1, 768]

                image_embeddings_list.append(img_feat)

            # C. Gộp đặc trưng Ảnh (Mean Pooling)
            # Chuyển list các [1, 768] thành tensor [N, 768]
            stacked_image_embeddings = torch.cat(image_embeddings_list, dim=0)
            # Lấy trung bình cộng theo chiều 0 (trung bình N ảnh)
            # và giữ lại chiều batch (keepdim=True)
            # Shape: [1, 768]
            agg_image_embedding = torch.mean(stacked_image_embeddings, dim=0, keepdim=True)

            # D. Gộp (Concat) Text và Ảnh đã gộp
            # [1, 768] + [1, 768] -> [1, 1536]
            combined_features = torch.cat((text_features, agg_image_embedding), dim=1)

            # E. Đưa qua đầu phân loại (Classifier Head)
            # [1, 1536] -> [1, num_classes]
            logits = self.model.classifier_head(combined_features)

        # 4. Tính xác suất (Áp dụng Sigmoid) - Giữ nguyên logic
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.cpu().numpy()[0]  # Lấy mảng 1D

        # 5. Quyết định nhãn - Giữ nguyên logic
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


# --- DÙNG ĐỂ CHẠY KIỂM TRA TRỰC TIẾP ---
if __name__ == "__main__":

    # --- CẤU HÌNH ĐỂ THỬ NGHIỆM ---
    # Sửa lại TEST_TEXT_CONTENT của bạn ở đây
    TEST_TEXT_CONTENT = ""

    # Kiểm tra xem config.DATA_DIR có tồn tại không
    if not os.path.exists(config.DATA_DIR):
        print(f"LỖI NGHIÊM TRỌNG: Thư mục DATA_DIR không tồn tại tại: {config.DATA_DIR}")
        print("Vui lòng kiểm tra lại file `src/config.py`")
        exit()

    # Thêm 3 đường dẫn ảnh để test
    TEST_IMAGE_PATHS = [
        os.path.join(config.DATA_DIR, 'raw/images/img.png'),
        os.path.join(config.DATA_DIR, 'raw/images/img_2.png'),
        os.path.join(config.DATA_DIR, 'raw/images/img_3.png')
    ]

    PREDICTION_THRESHOLD = 0.5
    # --------------------------------

    MODEL_PATH = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')

    # 1. Khởi tạo Predictor
    try:
        predictor = Predictor(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi khởi tạo Predictor: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 2. Chạy dự đoán
    print(f"\n--- Đang dự đoán cho ---")
    print(f"Text: {TEST_TEXT_CONTENT}")
    print(f"Ảnh (List):")

    # Kiểm tra xem ảnh nào tồn tại
    valid_image_paths = []
    for path in TEST_IMAGE_PATHS:
        if os.path.exists(path):
            print(f"  - [OK] {path}")
            valid_image_paths.append(path)
        else:
            # IN LỖI RÕ RÀNG CHO TỪNG FILE
            print(f"  - [LỖI] Không tìm thấy: {path}")

    if not valid_image_paths:
        print("\nLỖI: Không có ảnh test nào hợp lệ. Vui lòng thêm ảnh vào `data/test/`")
    else:
        # Chạy dự đoán với DANH SÁCH ảnh
        results = predictor.predict(
            text_content=TEST_TEXT_CONTENT,
            image_paths=valid_image_paths,
            threshold=PREDICTION_THRESHOLD
        )

        print("\n--- KẾT QUẢ DỰ ĐOÁN (cho 1 text, N ảnh) ---")
        print(f"(Với ngưỡng = {PREDICTION_THRESHOLD})")
        print(f"\nCác nhãn được dự đoán:")
        if results["predicted_labels"]:
            for label in results["predicted_labels"]:
                print(f"  - {label} (Score: {results['all_probabilities'][label]:.4f})")
        else:
            print("  (Không có nhãn nào vượt ngưỡng)")

        print("\n--- (Chi tiết tất cả xác suất) ---")
        sorted_probs = sorted(results["all_probabilities"].items(), key=lambda item: item[1], reverse=True)
        for label, prob in sorted_probs[:5]:
            print(f"  {label}: {prob:.4f}")