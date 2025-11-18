import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms
import numpy as np

# Import cấu hình từ file config.py
try:
    from src import config
except ImportError:
    print("Lỗi: Không thể import config.py. Đảm bảo file tồn tại và nằm trong cùng thư mục `src`.")
    exit()


class MultimodalDataset(Dataset):
    """
    Class Dataset tùy chỉnh cho bài toán phân loại đa phương thức (text, image) và đa nhãn (multi-label).
    """

    def __init__(self, csv_path, image_dir, labels_map, text_tokenizer, image_transform,
                 text_col, image_col, label_col, max_token_length):
        """
        Args:
            csv_path (str): Đường dẫn đến file train.csv, val.csv hoặc test.csv.
            image_dir (str): Đường dẫn đến thư mục chứa *tất cả* các ảnh.
            labels_map (dict): Dict map từ tên nhãn sang ID (ví dụ: {'game': 0, 'thể thao': 1}).
            text_tokenizer: Tokenizer của Hugging Face (ví dụ: PhoBERT tokenizer).
            image_transform: Các phép biến đổi (transform) của torchvision cho ảnh.
            text_col (str): Tên cột chứa text.
            image_col (str): Tên cột chứa tên file ảnh (image_path).
            label_col (str): Tên cột chứa nhãn.
            max_token_length (int): Độ dài tối đa để padding/truncate text.
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.labels_map = labels_map
        self.num_classes = len(labels_map)
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform

        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.max_token_length = max_token_length

        # Xử lý các giá trị NaN (nếu có)
        self.df[self.text_col] = self.df[self.text_col].fillna('').astype(str)
        self.df[self.label_col] = self.df[self.label_col].fillna('').astype(str)

    def __len__(self):
        """Trả về số lượng mẫu trong dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu tại vị trí idx."""

        # 1. Lấy dữ liệu thô từ DataFrame
        row = self.df.iloc[idx]
        raw_text = row[self.text_col]
        image_filename = row[self.image_col]
        label_str = row[self.label_col]

        # 2. Xử lý Văn bản (Text)
        text_inputs = self.text_tokenizer(
            raw_text,
            max_length=self.max_token_length,
            padding='max_length',  # Pad đến max_length
            truncation=True,  # Cắt nếu dài hơn max_length
            return_tensors='pt'  # Trả về PyTorch tensors
        )

        # `squeeze()` để bỏ bớt chiều batch (ví dụ: [1, 128] -> [128])
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # 3. Xử lý Ảnh (Image)
        image_path = os.path.join(self.image_dir, image_filename)
        image = None

        try:
            # Mở file ở chế độ 'rb' (read binary)
            with open(image_path, 'rb') as f:
                # đọc nội dung file (bytes)
                image_pil = Image.open(f)

                # .convert('RGB')
                image = image_pil.convert('RGB')

        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy ảnh {image_path}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')
        except Exception as e:
            # Bất kỳ lỗi nào khác (file hỏng, không phải định dạng ảnh...)
            print(f"LỖI khi đọc ảnh {image_path}: {e}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')

        # Áp dụng các phép biến đổi (resize, to_tensor, normalize)
        image_tensor = self.image_transform(image)
        # -----------------------------------------------

        # 4. Xử lý Nhãn (Label) -> Tạo Vector Multi-hot
        # Bắt đầu với một vector toàn số 0, kiểu float
        label_vector = torch.zeros(self.num_classes, dtype=torch.float)

        # Tách chuỗi nhãn
        if label_str.strip():
            labels = [label.strip() for label in label_str.split(',')]
            for label in labels:
                if label in self.labels_map:
                    # Lấy ID của nhãn
                    label_id = self.labels_map[label]
                    label_vector[label_id] = 1.0


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_tensor': image_tensor,
            'labels': label_vector
        }


# --- Các hàm (helper function) để tạo Dataset và DataLoader ---

def get_image_transforms():
    """Trả về các phép biến đổi chuẩn cho ViT."""
    # ViT chuẩn hóa (normalize)
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def create_data_loader(split='train'):
    """
    Hàm chính để tạo DataLoader cho một 'split' (train/val/test).
    """
    if split == 'train':
        csv_name = 'train.csv'
    elif split == 'val':
        csv_name = 'val.csv'
    elif split == 'test':
        csv_name = 'test.csv'
    else:
        raise ValueError(f"Split '{split}' không hợp lệ. Phải là 'train', 'val', hoặc 'test'.")

    csv_path = os.path.join(config.SPLITS_DIR, csv_name)

    # 1. Tải Tokenizer (cho text)
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    except Exception as e:
        print(f"Lỗi khi tải Tokenizer {config.TEXT_MODEL_NAME}: {e}")
        print("Vui lòng kiểm tra lại tên model và kết nối mạng.")
        exit()

    # 2. Lấy Image Transforms (cho ảnh)
    image_transform = get_image_transforms()

    # 3. Tải Bản đồ nhãn (labels_map)
    try:
        with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {config.LABELS_MAP_PATH}")
        print("Vui lòng chạy lại script `split_data.py` (Bước 3).")
        exit()

    # 4. Khởi tạo Dataset
    dataset = MultimodalDataset(
        csv_path=csv_path,
        image_dir=config.IMAGE_DIR,
        labels_map=labels_map,
        text_tokenizer=text_tokenizer,
        image_transform=image_transform,
        text_col=config.TEXT_COLUMN,
        image_col=config.IMAGE_COLUMN,
        label_col=config.LABEL_COLUMN,
        max_token_length=config.MAX_TOKEN_LENGTH
    )

    # 5. Khởi tạo DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == 'train'),  # Chỉ xáo trộn (shuffle) khi train
        num_workers=os.cpu_count() // 2  # Sử dụng nhiều CPU để tải dữ liệu
    )

    return data_loader, dataset



if __name__ == "__main__":
    print("--- KIỂM TRA BƯỚC 4: TẠO DATA LOADER ---")
    print(f"Đang tải cấu hình từ config.py...")
    print(f"IMAGE_DIR: {config.IMAGE_DIR}")
    print(f"LABELS_MAP_PATH: {config.LABELS_MAP_PATH}")
    print(f"SPLITS_DIR: {config.SPLITS_DIR}")

    print("\nĐang thử tạo DataLoader cho tập 'train'...")
    try:
        train_loader, train_dataset = create_data_loader('train')
        print(f"\nTạo DataLoader thành công!")
        print(f"Số lượng mẫu trong tập train: {len(train_dataset)}")
        print(f"Số lượng lớp (nhãn): {train_dataset.num_classes}")

        # Lấy 1 batch đầu tiên để kiểm tra
        print("\nĐang lấy 1 batch dữ liệu...")
        first_batch = next(iter(train_loader))

        print("Kiểm tra kích thước (shape) của batch đầu tiên:")
        print(f"  - input_ids (text): {first_batch['input_ids'].shape}")
        print(f"  - attention_mask (text): {first_batch['attention_mask'].shape}")
        print(f"  - image_tensor (ảnh): {first_batch['image_tensor'].shape}")
        print(f"  - labels (vector nhãn): {first_batch['labels'].shape}")

        print("\nKiểm tra một vector nhãn (multi-hot):")
        print(first_batch['labels'][0])

        print("\n--- HOÀN THÀNH KIỂM TRA BƯỚC 4 ---")

    except Exception as e:
        print(f"\n--- LỖI TRONG QUÁ TRÌNH KIỂM TRA ---")
        import traceback

        traceback.print_exc()
        print("\n>>> Vui lòng kiểm tra lại các đường dẫn trong `config.py` và file `labels_map.json`.")