import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os

# Sửa lỗi Matplotlib (nếu có)
# Phải import 2 dòng này *TRƯỚC* khi import plt từ torchvision
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import các file theo cách của PyCharm
try:
    from src import config
    from src.exp_v2_cnn.ip_utils import (
        otsu_threshold, morph_opening, morph_closing, custom_rgb_to_grayscale
    )
    from src.exp_v2_cnn.model_v2 import SobelConv
except ImportError:
    print("Lỗi import. Hãy chạy file này bằng PyCharm hoặc từ thư mục gốc ml_service.")
    exit()

# === CẤU HÌNH ===
TEST_IMAGE_NAME = 'aimg_1.png'


# ==================

def run_visualization():
    # 1. Tải ảnh gốc
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TEST_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'images', TEST_IMAGE_NAME)

    try:
        img_pil = Image.open(TEST_IMAGE_PATH).convert('RGB')
        img_pil_resized = img_pil.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file ảnh tại: {TEST_IMAGE_PATH}")
        return

    # --- CHẠY TỪNG BƯỚC XỬ LÝ ẢNH ---

    # 2. Bước 1: Grayscale (từ ip_utils)
    img_gray_np = custom_rgb_to_grayscale(np.array(img_pil_resized))

    # 3. Bước 2: Otsu (từ ip_utils)
    threshold = otsu_threshold(img_gray_np)
    img_binary_np = (img_gray_np > threshold)  # Ảnh True/False

    # 4. Bước 3: Opening + Closing (từ ip_utils)
    img_opened_np = morph_opening(img_binary_np, kernel_size=3)
    img_closed_np = morph_closing(img_opened_np, kernel_size=3)  # Đây là input cho CNN

    # 5. Bước 4: SobelConv (từ model_v2)
    # Khởi tạo Lớp Sobel
    sobel_layer = SobelConv(in_channels=1, freeze=True)

    # Chuyển ảnh "sạch" (đã Closing) sang Tensor
    # Phải chuyển (True/False) sang (1.0/0.0)
    binary_tensor_input = torch.from_numpy(img_closed_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Đẩy ảnh qua Lớp Sobel
    with torch.no_grad():
        sobel_tensor_output = sobel_layer(binary_tensor_input)

    img_sobel_np = sobel_tensor_output.squeeze().numpy()

    # 6. Vẽ kết quả (4 bước)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Trực quan hóa Pipeline V2 (Đầy đủ): {TEST_IMAGE_NAME}", fontsize=16)

    axes[0, 0].imshow(img_gray_np, cmap='gray')
    axes[0, 0].set_title('1. Ảnh Grayscale (Đầu vào Otsu)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_binary_np, cmap='gray')
    axes[0, 1].set_title(f'2. Ảnh Nhị phân (Sau Otsu T={threshold})')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img_closed_np, cmap='gray')
    axes[1, 0].set_title('3. Ảnh "Sạch" (Sau Opening + Closing)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_sobel_np, cmap='gray')
    axes[1, 1].set_title('4. Ảnh "Cạnh" (Sau SobelConv - Đầu vào CNN)')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_visualization()