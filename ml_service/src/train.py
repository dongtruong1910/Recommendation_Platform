import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import json

# Import các file khác của chúng ta
try:
    from src import config
    from src.dataset import create_data_loader
    from src.model import get_model  # Chúng ta dùng hàm helper này
except ImportError:
    print("Lỗi: Không thể import config, dataset, hoặc model.")
    print("Hãy đảm bảo bạn đang chạy file này từ thư mục gốc `ml_service` và đã export PYTHONPATH.")
    exit()


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Huấn luyện mô hình trong 1 epoch."""
    model.train()  # Đặt mô hình ở chế độ training
    total_loss = 0.0

    # Dùng tqdm để tạo thanh tiến trình
    progress_bar = tqdm(data_loader, desc="Epoch Train", leave=False)

    for batch in progress_bar:
        # 1. Di chuyển dữ liệu lên GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_tensor = batch['image_tensor'].to(device)
        labels = batch['labels'].to(device)

        # 2. Xóa gradient cũ
        optimizer.zero_grad()

        # 3. Forward pass (Đưa dữ liệu qua mô hình)
        logits = model(input_ids, attention_mask, image_tensor)

        # 4. Tính loss
        # BCEWithLogitsLoss yêu cầu logits (raw) và labels (dạng float)
        loss = loss_fn(logits, labels)

        # 5. Backward pass (Lan truyền ngược)
        loss.backward()

        # 6. Cập nhật trọng số
        optimizer.step()

        total_loss += loss.item()

        # Cập nhật thanh tiến trình
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def evaluate(model, data_loader, loss_fn, device):
    """Đánh giá mô hình trên tập validation."""
    model.eval()  # Đặt mô hình ở chế độ evaluation (tắt dropout,...)
    total_loss = 0.0

    # Không cần tính gradient khi đánh giá
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation", leave=False)

        for batch in progress_bar:
            # 1. Di chuyển dữ liệu lên GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            labels = batch['labels'].to(device)

            # 2. Forward pass
            logits = model(input_ids, attention_mask, image_tensor)

            # 3. Tính loss
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def run_training():
    """Hàm chính để chạy toàn bộ quá trình huấn luyện."""

    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ---")

    # 1. Thiết lập Device (GPU)
    # Đây là cách chuẩn để kiểm tra và sử dụng GPU (RTX 4060)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    if device.type == 'cpu':
        print("CẢNH BÁO: Không tìm thấy GPU. Huấn luyện trên CPU sẽ RẤT CHẬM.")

    # 2. Tải Dữ liệu
    print("Đang tải DataLoaders...")
    train_loader, _ = create_data_loader('train')
    val_loader, val_dataset = create_data_loader('val')
    print("Tải dữ liệu thành công.")

    # 3. Khởi tạo Mô hình
    print("Đang khởi tạo mô hình...")
    # (Việc tải ViT và PhoBERT có thể mất vài phút nếu là lần đầu tiên)
    # Lưu ý: get_model() đang trả về mô hình với backbone đã bị đóng băng
    model = get_model()
    model.to(device)  # Di chuyển mô hình lên GPU
    print("Khởi tạo mô hình thành công.")

    # 4. Định nghĩa Loss và Optimizer
    # Hàm loss chuẩn cho Multi-label
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer: AdamW (tốt cho Transformer)
    # Quan trọng: Vì chúng ta đóng băng backbone, chúng ta chỉ train các tham số
    # của 'classifier_head' (là các tham số có requires_grad=True)
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    print(f"Số lượng tham số cần huấn luyện (chỉ classifier head): {sum(p.numel() for p in params_to_train)}")

    optimizer = AdamW(params_to_train, lr=config.LEARNING_RATE)

    # 5. Vòng lặp Huấn luyện
    best_val_loss = float('inf')  # Đặt loss tốt nhất là vô cực

    # Tạo thư mục để lưu model
    model_save_dir = os.path.join(config.BASE_DIR, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, 'best_model.pth')

    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(config.EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {config.EPOCHS} ========")

        # Huấn luyện
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"  Trung bình Train Loss: {train_loss:.4f}")

        # Đánh giá
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"  Trung bình Validation Loss: {val_loss:.4f}")

        # Lưu lại mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ==> Validation Loss cải thiện. Đã lưu mô hình tốt nhất tại: {best_model_path}")
        else:
            print("  Validation Loss không cải thiện.")

    print("\n--- HOÀN THÀNH HUẤN LUYỆN ---")
    print(f"Mô hình tốt nhất đã được lưu tại: {best_model_path}")


# Chạy hàm chính
if __name__ == "__main__":
    run_training()