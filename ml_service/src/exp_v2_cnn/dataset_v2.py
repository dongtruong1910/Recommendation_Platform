import os
import json
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms
from src import config
from src.exp_v2_cnn.ip_utils import otsu_threshold, morph_opening, morph_closing



class MultimodalDatasetV2(Dataset):
    def __init__(self, csv_path, image_dir, labels_map, text_tokenizer,
                 text_col, image_col, label_col, max_token_length):

        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.labels_map = labels_map
        self.num_classes = len(labels_map)
        self.text_tokenizer = text_tokenizer

        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.max_token_length = max_token_length

        self.df[self.text_col] = self.df[self.text_col].fillna('').astype(str)
        self.df[self.label_col] = self.df[self.label_col].fillna('').astype(str)

        self.image_transform_v2 = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(),
            np.array,
            transforms.Lambda(self.apply_ip_pipeline),
            transforms.ToTensor(),
        ])

    def apply_ip_pipeline(self, image_gray_numpy):
        threshold = otsu_threshold(image_gray_numpy)
        binary_image = image_gray_numpy > threshold
        cleaned_image = morph_opening(binary_image, kernel_size=3)
        cleaned_image = morph_closing(cleaned_image, kernel_size=3) # Tùy chọn
        return cleaned_image.astype(np.uint8) * 255

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Text
        raw_text = row[self.text_col]
        text_inputs = self.text_tokenizer(
            raw_text,
            max_length=self.max_token_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # 2. Image
        image_filename = row[self.image_col]
        if isinstance(image_filename, str):
            image_filename = os.path.basename(image_filename)
        else:
            image_filename = ""

        image_path = os.path.join(self.image_dir, image_filename)

        try:
            with open(image_path, 'rb') as f:
                image_pil = Image.open(f)
                image = image_pil.convert('RGB')
        except Exception:
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')

        image_tensor = self.image_transform_v2(image)

        # 3. Label
        label_str = row[self.label_col]
        label_vector = torch.zeros(self.num_classes, dtype=torch.float)

        if label_str.strip():
            labels = [label.strip() for label in label_str.split(',')]
            for label in labels:
                if label in self.labels_map:
                    label_id = self.labels_map[label]
                    label_vector[label_id] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_tensor': image_tensor,
            'labels': label_vector
        }



def create_data_loader_v2(split='train'):
    if split == 'train':
        csv_name = 'train.csv'
    elif split == 'val':
        csv_name = 'val.csv'
    elif split == 'test':
        csv_name = 'test.csv'
    else:
        raise ValueError(f"Split '{split}' không hợp lệ.")

    csv_path = os.path.join(config.SPLITS_DIR, csv_name)

    text_tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)

    dataset = MultimodalDatasetV2(
        csv_path=csv_path,
        image_dir=config.IMAGE_DIR,
        labels_map=labels_map,
        text_tokenizer=text_tokenizer,
        text_col=config.TEXT_COLUMN,
        image_col=config.IMAGE_COLUMN,
        label_col=config.LABEL_COLUMN,
        max_token_length=config.MAX_TOKEN_LENGTH
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == 'train'),
        num_workers=os.cpu_count() // 2
    )

    return data_loader, dataset



if __name__ == "__main__":
    print("--- KIỂM TRA BƯỚC 2: TẠO dataset_v2.py (đã sửa import) ---")

    print("\nĐang thử tạo DataLoader V2 cho tập 'train'...")
    try:
        train_loader, train_dataset = create_data_loader_v2('train')
        print(f"\nTạo DataLoader thành công!")
        print(f"Số lượng mẫu trong tập train: {len(train_dataset)}")

        print("\nĐang lấy 1 batch dữ liệu...")
        first_batch = next(iter(train_loader))

        print("Kiểm tra kích thước (shape) của batch đầu tiên:")
        print(f"  - input_ids (text): {first_batch['input_ids'].shape}")
        print(f"  - attention_mask (text): {first_batch['attention_mask'].shape}")
        print(f"  - image_tensor (ảnh): {first_batch['image_tensor'].shape}")
        print(f"  - labels (vector nhãn): {first_batch['labels'].shape}")

        expected_image_shape = (config.BATCH_SIZE, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        if first_batch['image_tensor'].shape == expected_image_shape:
            print(f"\nKích thước ảnh {expected_image_shape} là đúng (1 kênh màu).")
        else:
            print(
                f"\nLỖI: Kích thước ảnh {first_batch['image_tensor'].shape} sai, lẽ ra phải là {expected_image_shape}.")

        print("\n--- HOÀN THÀNH KIỂM TRA BƯỚC 2 ---")

    except Exception as e:
        print(f"\n--- LỖI TRONG QUÁ TRÌNH KIỂM TRA ---")
        import traceback

        traceback.print_exc()