import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import json

try:
    from src import config
    from src.exp_v2_cnn.dataset_v2 import create_data_loader_v2
    from src.exp_v2_cnn.model_v2 import get_model_v2
except ImportError:
    # Thử import theo cách "module" (nếu chạy bằng -m)
    from .. import config
    from .dataset_v2 import create_data_loader_v2
    from .model_v2 import get_model_v2


# ==========================================================
# ===TRAIN/EVAL ===
# ==========================================================

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Huấn luyện mô hình trong 1 epoch."""
    model.train()  # Đặt mô hình ở chế độ training
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Epoch Train V2", leave=False)

    for batch in progress_bar:
        # 1. Di chuyển dữ liệu lên GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_tensor = batch['image_tensor'].to(device)  # Ảnh 1 kênh (Otsu)
        labels = batch['labels'].to(device)

        # 2. Xóa gradient cũ
        optimizer.zero_grad()

        # 3. Forward pass
        logits = model(input_ids, attention_mask, image_tensor)

        # 4. Tính loss
        loss = loss_fn(logits, labels)

        # 5. Backward pass
        loss.backward()

        # 6. Cập nhật trọng số
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def evaluate(model, data_loader, loss_fn, device):
    """Đánh giá mô hình trên tập validation."""
    model.eval()  # Đặt mô hình ở chế độ evaluation
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation V2", leave=False)

        for batch in progress_bar:
            # 1. Di chuyển dữ liệu lên GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor = batch['image_tensor'].to(device)  # Ảnh 1 kênh (Otsu)
            labels = batch['labels'].to(device)

            # 2. Forward pass
            logits = model(input_ids, attention_mask, image_tensor)

            # 3. Tính loss
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


# ==========================================================
# === RUN ===
# ==========================================================

def run_training_v2():
    """Hàm chính để chạy toàn bộ quá trình huấn luyện V2."""

    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN V2 (CustomCNN + PhoBERT) ---")

    # 1. Thiết lập Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    if device.type == 'cpu':
        print("CẢNH BÁO: Không tìm thấy GPU. Huấn luyện trên CPU sẽ RẤT CHẬM.")

    # 2. Tải Dữ liệu
    print("Đang tải DataLoaders V2 (Otsu/Morph)...")
    train_loader, _ = create_data_loader_v2('train')
    val_loader, _ = create_data_loader_v2('val')
    print("Tải dữ liệu V2 thành công.")

    # 3. Khởi tạo Mô hình
    print("Đang khởi tạo mô hình V2 (get_model_v2)...")
    model_v2 = get_model_v2()
    model_v2.to(device)  # Di chuyển mô hình lên GPU
    print("Khởi tạo mô hình V2 thành công.")

    # 4. Định nghĩa Loss và Optimizer
    loss_fn = nn.BCEWithLogitsLoss()

    # đóng băng PhoBERT, train CustomCNN VÀ classifier_head.
    # Các tham số này đã được thiết lập requires_grad=True trong model_v2.py
    params_to_train_v2 = [p for p in model_v2.parameters() if p.requires_grad]

    print(f"Số lượng tham số cần huấn luyện (CustomCNN + Head): {sum(p.numel() for p in params_to_train_v2)}")

    # Dùng learning rate từ config
    optimizer = AdamW(params_to_train_v2, lr=config.LEARNING_RATE)

    # 5. Vòng lặp Huấn luyện
    best_val_loss = float('inf')

    # lưu model
    model_save_dir = os.path.join(config.BASE_DIR, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    #file lưu model tốt nhất
    best_model_v2_path = os.path.join(model_save_dir, 'best_model_v2_cnn.pth')

    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN V2 (sẽ lưu tại {best_model_v2_path}) ---")
    for epoch in range(config.EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {config.EPOCHS} (V2) ========")

        # Huấn luyện
        train_loss = train_one_epoch(model_v2, train_loader, loss_fn, optimizer, device)
        print(f"  Trung bình Train Loss V2: {train_loss:.4f}")

        # Đánh giá
        val_loss = evaluate(model_v2, val_loader, loss_fn, device)
        print(f"  Trung bình Validation Loss V2: {val_loss:.4f}")

        # Lưu lại mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_v2.state_dict(), best_model_v2_path)
            print(f"  ==> Validation Loss V2 cải thiện. Đã lưu mô hình tại: {best_model_v2_path}")
        else:
            print("  Validation Loss V2 không cải thiện.")

    print("\n--- HOÀN THÀNH HUẤN LUYỆN V2 ---")
    print(f"Mô hình V2 tốt nhất đã được lưu tại: {best_model_v2_path}")


# Chạy hàm chính
if __name__ == "__main__":
    run_training_v2()