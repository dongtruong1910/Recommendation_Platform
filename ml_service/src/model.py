import torch
import torch.nn as nn
from transformers import AutoModel
import json
import os

# Import cấu hình từ file config.py
try:
    from src import config
except ImportError:
    print("Lỗi: Không thể import config.py. Đảm bảo file tồn tại và nằm trong cùng thư mục `src`.")
    exit()


class MultimodalClassifier(nn.Module):
    """
    Mô hình phân loại đa phương thức (Text + Image) và đa nhãn (Multi-label).
    Kết hợp PhoBERT và ViT.
    """

    def __init__(self, num_classes, freeze_backbones=False):
        """
        Args:
            num_classes (int): Số lượng nhãn đầu ra.
            freeze_backbones (bool): Nếu True, đóng băng trọng số của ViT và PhoBERT.
        """
        super(MultimodalClassifier, self).__init__()

        self.num_classes = num_classes

        print(f"Đang tải mô hình pre-trained {config.TEXT_MODEL_NAME}...")
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME, use_safetensors=True)

        print(f"Đang tải mô hình pre-trained {config.IMAGE_MODEL_NAME}...")
        self.image_model = AutoModel.from_pretrained(config.IMAGE_MODEL_NAME)

        # Lấy kích thước đặc trưng (768 cho cả 2)
        self.text_feature_dim = self.text_model.config.hidden_size
        self.image_feature_dim = self.image_model.config.hidden_size
        combined_feature_dim = self.text_feature_dim + self.image_feature_dim  # 768 + 768 = 1536

        print(f"Kích thước đặc trưng Text: {self.text_feature_dim}")
        print(f"Kích thước đặc trưng Image: {self.image_feature_dim}")
        print(f"Kích thước đặc trưng kết hợp: {combined_feature_dim}")

        # Đóng băng (Freeze)
        if freeze_backbones:
            print("Đang đóng băng trọng số của ViT và PhoBERT...")
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.image_model.parameters():
                param.requires_grad = False

        # 4. Đầu phân loại (Classifier Head)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(combined_feature_dim),  # Chuẩn hóa đặc trưng
            nn.Linear(combined_feature_dim, 512),
            nn.GELU(),  # Hàm kích hoạt
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, input_ids, attention_mask, image_tensor):
        """
        Phép lan truyền thuận (forward pass).
        """

        # 1. Nhánh Văn bản (Text)
        # Lấy [CLS] token's last hidden state
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # text_features = text_outputs.last_hidden_state[:, 0, :] # Lấy [CLS] token
        text_features = text_outputs.pooler_output

        # 2. Nhánh Hình ảnh (Image)
        image_outputs = self.image_model(
            pixel_values=image_tensor
        )
        # image_features = image_outputs.last_hidden_state[:, 0, :] # Lấy [CLS] token
        image_features = image_outputs.pooler_output

        # 3. Kết hợp (Fusion)
        # Ghép 2 vector đặc trưng lại
        # Kích thước: [batch_size, 768+768]
        combined_features = torch.cat((text_features, image_features), dim=1)

        # 4. Đưa qua đầu phân loại
        # Kích thước: [batch_size, num_classes]
        logits = self.classifier_head(combined_features)

        return logits


def get_model():
    """Hàm helper để tải mô hình."""
    # 1. Lấy số lượng nhãn từ file labels_map.json
    try:
        with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)
        num_classes = len(labels_map)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {config.LABELS_MAP_PATH}")
        print("Vui lòng chạy lại script `split_data.py` (Bước 3).")
        exit()

    # 2. Khởi tạo mô hình
    model = MultimodalClassifier(num_classes=num_classes, freeze_backbones=True)

    return model


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from dataset import create_data_loader
    except ImportError:
        print("Lỗi: Không thể import `create_data_loader` từ `dataset.py`.")
        print("Hãy đảm bảo bạn đang chạy file này từ thư mục gốc `ml_service`.")
        exit()

    print("\n--- KIỂM TRA BƯỚC 5: XÂY DỰNG MÔ HÌNH ---")

    # 1. Tải 1 batch dữ liệu từ DataLoader
    print("Đang tải 1 batch dữ liệu (val) để kiểm tra...")
    try:
        # Dùng 'val' để kiểm tra nhanh hơn, không cần shuffle
        val_loader, val_dataset = create_data_loader('val')
        first_batch = next(iter(val_loader))
        num_classes = val_dataset.num_classes
        print(f"Tải batch thành công. Số lượng nhãn (num_classes): {num_classes}")
    except Exception as e:
        print(f"LỖI khi tải DataLoader: {e}")
        exit()

    # 2. Khởi tạo mô hình
    print("\nĐang khởi tạo mô hình MultimodalClassifier...")
    model = MultimodalClassifier(num_classes=num_classes, freeze_backbones=True)
    print("Khởi tạo mô hình thành công!")

    # 3. Đẩy dữ liệu qua mô hình
    print("\nĐang thực hiện forward pass...")
    # Lấy dữ liệu từ batch
    input_ids = first_batch['input_ids']
    attention_mask = first_batch['attention_mask']
    image_tensor = first_batch['image_tensor']


    with torch.no_grad():
        logits = model(input_ids, attention_mask, image_tensor)

    print("Forward pass thành công!")

    # 4. Kiểm tra kích thước đầu ra
    print(f"\nKích thước Batch Size: {config.BATCH_SIZE}")
    print(f"Kích thước Logits đầu ra: {logits.shape}")

    expected_shape = (config.BATCH_SIZE, num_classes)
    if logits.shape == expected_shape:
        print(f"Kích thước đầu ra {logits.shape} khớp với mong đợi {expected_shape}.")
        print("\n--- HOÀN THÀNH KIỂM TRA BƯỚC 5 ---")
    else:
        print(f"LỖI: Kích thước đầu ra {logits.shape} KHÔNG khớp với mong đợi {expected_shape}.")