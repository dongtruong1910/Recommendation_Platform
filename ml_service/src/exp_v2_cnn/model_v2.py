import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
import json
import os

try:
    from src import config
    from src.exp_v2_cnn.dataset_v2 import create_data_loader_v2
except ImportError:
    # Thử import theo cách "module" (nếu chạy bằng -m)
    from .. import config
    from .dataset_v2 import create_data_loader_v2


# ==========================================================
# === ÁP DỤNG KIẾN THỨC XỬ LÝ ẢNH ===
# ==========================================================
class SobelConv(nn.Module):
    """
    Lớp Conv2d tùy chỉnh, được dùng trọng số Sobel.
    """

    def __init__(self, in_channels=1, freeze=False):
        super(SobelConv, self).__init__()

        # Chúng ta sẽ tạo 2 kernel (bộ lọc)
        self.in_channels = in_channels
        self.out_channels = in_channels * 2  # 1 kernel cho Gx, 1 cho Gy

        # Khởi tạo một lớp Conv2d 2D bình thường
        # Kernel 3x3, padding 1 để giữ nguyên kích thước ảnh
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=3, padding=1, bias=False,
                              groups=self.in_channels)  # groups=in_channels để áp Gx, Gy riêng

        # --- Dùng trọng số Sobel ---
        # 1. Tạo kernel Sobel (dạng numpy)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # 2. Chuyển sang dạng Tensor
        # Kích thước chuẩn của Pytorch: [out_channels, in_channels/groups, H, W]
        # Kích thước của ta: [2, 1, 3, 3]
        weight = torch.zeros((self.out_channels, 1, 3, 3))
        weight[0, 0] = torch.from_numpy(sobel_x)  # Gx
        weight[1, 0] = torch.from_numpy(sobel_y)  # Gy

        # 3. Gán trọng số đã dùng Sobel vào lớp Conv
        self.conv.weight.data.copy_(weight)

        if freeze:
            print("Đóng băng lớp Sobel.")
            self.conv.weight.requires_grad = False

    def forward(self, x):
        # x có shape [B, 1, 224, 224]

        # 1. Chạy qua lớp Conv đã dùng Sobel
        x = self.conv(x)  # Shape [B, 2, 224, 224] (1 kênh Gx, 1 kênh Gy)

        # 2. Tính toán độ lớn (Magnitude)
        # Tương tự như sqrt(Gx^2 + Gy^2)
        # Dùng .pow(2).sum(dim=1) (tổng bình phương 2 kênh)
        # .sqrt() và thêm 1e-6 để tránh lỗi chia cho 0
        x = (x.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
        # Shape [B, 1, 224, 224] (Ảnh cạnh)

        return x


# ==========================================================
# === MÔ HÌNH CNN ===
# ==========================================================
class CustomCNN(nn.Module):
    """
    Mô hình CNN.
    """

    def __init__(self, in_channels=1, output_dim=768, use_sobel_layer=True):
        super(CustomCNN, self).__init__()

        current_channels = in_channels

        # --- Lớp 1: Lớp Xử lý ảnh (Tùy chọn) ---
        if use_sobel_layer:
            # Dùng lớp Sobel tùy chỉnh ở trên
            # và đóng băng nó (freeze=True)
            self.ip_layer = SobelConv(in_channels=in_channels, freeze=True)
            print("Đã khởi tạo Lớp Sobel (bị đóng băng).")
            # Kênh đầu ra của Sobel vẫn là 1 (sau khi tính magnitude)
            current_channels = 1
        else:
            # Dùng 1 lớp Conv bình thường (tự học từ đầu)
            self.ip_layer = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
            current_channels = 16
            print("Đã khởi tạo Lớp Conv đầu tiên (học từ đầu).")

        # --- Các Khối (Blocks) CNN ---
        # Một khối = Conv -> BatchNorm -> Relu -> MaxPool

        # Khối 1
        self.conv1 = nn.Conv2d(current_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Giảm kích thước 224 -> 112

        # Khối 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Giảm kích thước 112 -> 56

        # Khối 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Giảm kích thước 56 -> 28

        # Khối 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # <-- NÂNG CẤP
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14

        # --- "Đầu" (Head) của CNN ---
        # Dùng AdaptiveAvgPool2d để ép 28x28 xuống 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Lớp Linear cuối cùng để ra vector đặc trưng
        self.fc = nn.Linear(256, output_dim)  # output_dim = 768

    def forward(self, x):
        # x có shape [B, 1, 224, 224]

        # 1. Qua lớp "IP Layer" (Sobel hoặc Conv thường)
        x = F.relu(self.ip_layer(x))

        # 2. Qua các khối CNN
        x = self.pool1(F.relu(self.conv1(x)))  # Shape [B, 32, 112, 112]
        x = self.pool2(F.relu(self.conv2(x)))  # Shape [B, 64, 56, 56]
        x = self.pool3(F.relu(self.conv3(x)))  # Shape [B, 128, 28, 28]
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # Shape [B, 256, 14, 14]

        # 3. Dàn phẳng (Flatten)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Shape [B, 256]

        # 4. Qua lớp Linear cuối
        image_features = self.fc(x)  # Shape [B, 768]

        return image_features


# ==========================================================
# === KẾT HỢP V2 (PhoBERT + CNN) ===
# ==========================================================
class MultimodalClassifierV2(nn.Module):
    def __init__(self, num_classes, use_sobel_layer=True):
        super(MultimodalClassifierV2, self).__init__()

        self.num_classes = num_classes

        # 1. Nhánh Text: Giữ nguyên PhoBERT (và đóng băng nó)
        print("Đang tải PhoBERT (đã đóng băng)...")
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME, use_safetensors=True)
        self.text_feature_dim = self.text_model.config.hidden_size  # 768
        # Đóng băng PhoBERT
        for param in self.text_model.parameters():
            param.requires_grad = False

        # 2. Nhánh Ảnh: Dùng CustomCNN
        print("Đang xây dựng CustomCNN...")
        # Đầu vào là 1 kênh (ảnh nhị phân từ Otsu)
        # Đầu ra là 768 (để khớp với PhoBERT)
        self.image_model = CustomCNN(in_channels=1, output_dim=768, use_sobel_layer=use_sobel_layer)
        self.image_feature_dim = 768  # 768

        # 3. Đầu phân loại (Classifier Head):
        combined_feature_dim = self.text_feature_dim + self.image_feature_dim  # 768 + 768 = 1536

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(combined_feature_dim),
            nn.Linear(combined_feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes)
        )

        print("Khởi tạo mô hình V2 (PhoBERT + CustomCNN) thành công.")

    def forward(self, input_ids, attention_mask, image_tensor):
        # 1. Nhánh Text (Đã đóng băng)
        with torch.no_grad():  # Tắt tính gradient cho PhoBERT
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.pooler_output

        # 2. Nhánh Ảnh (Sẽ được train)
        image_features = self.image_model(image_tensor)

        # 3. Kết hợp
        combined_features = torch.cat((text_features, image_features), dim=1)

        # 4. Đầu phân loại (Sẽ được train)
        logits = self.classifier_head(combined_features)

        return logits


def get_model_v2():
    """Hàm helper để tải mô hình V2."""
    with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)
    num_classes = len(labels_map)

    # dùng Lớp Sobel
    model = MultimodalClassifierV2(num_classes=num_classes, use_sobel_layer=True)
    return model



if __name__ == "__main__":
    print("\n--- KIỂM TRA BƯỚC 3: TẠO model_v2.py ---")

    # 1. Tải 1 batch dữ liệu từ DataLoader V2
    print("Đang tải 1 batch dữ liệu V2 (val) để kiểm tra...")
    try:
        val_loader, val_dataset = create_data_loader_v2('val')
        first_batch = next(iter(val_loader))
        num_classes = val_dataset.num_classes
        print(f"Tải batch V2 thành công. Số lượng nhãn: {num_classes}")
    except Exception as e:
        print(f"LỖI khi tải DataLoader V2: {e}")
        import traceback

        traceback.print_exc()
        exit()

    # 2. Khởi tạo mô hình V2
    print("\nĐang khởi tạo mô hình MultimodalClassifierV2...")
    model_v2 = get_model_v2()

    # 3. Đẩy dữ liệu qua mô hình
    print("\nĐang thực hiện forward pass V2...")
    input_ids = first_batch['input_ids']
    attention_mask = first_batch['attention_mask']
    image_tensor = first_batch['image_tensor']  # Đây là ảnh 1 kênh (Otsu)

    with torch.no_grad():
        logits = model_v2(input_ids, attention_mask, image_tensor)

    print("Forward pass V2 thành công!")

    # 4. Kiểm tra kích thước đầu ra
    print(f"\nKích thước Batch Size: {config.BATCH_SIZE}")
    print(f"Kích thước Logits đầu ra: {logits.shape}")

    expected_shape = (config.BATCH_SIZE, num_classes)
    if logits.shape == expected_shape:
        print(f"Kích thước đầu ra {logits.shape} khớp với mong đợi {expected_shape}.")
        print("\n--- HOÀN THÀNH KIỂM TRA BƯỚC 3 ---")
    else:
        print(f"LỖI: Kích thước đầu ra {logits.shape} KHÔNG khớp với mong đợi {expected_shape}.")