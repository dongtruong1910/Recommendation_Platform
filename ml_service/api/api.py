# ========================
# IMPORT MODULES
# ========================
from urllib.parse import urlparse

import torch
import json
import requests
import tempfile
import uuid
import os  #

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any


try:
    from src import config
    from src.predict import Predictor
except ImportError as e:
    print("\n🔥 LỖI IMPORT SRC !!!")
    print("Lỗi này xảy ra nếu bạn bấm 'Play' trực tiếp trên file này.")
    print("Hãy chạy bằng Cấu hình Run 'uvicorn' của PyCharm hoặc lệnh terminal.")
    print("Chi tiết:", e)
    raise

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ========================
# DATA MODELS
# ========================

class PostInput(BaseModel):
    post_id: str
    text_content: str
    image_urls: List[str] = []

class BatchInput(BaseModel):
    posts: List[PostInput]

class PredictionResult(BaseModel):
    predicted_labels: List[str]
    all_probabilities: Dict[str, float]

class BatchResponseItem(BaseModel):
    post_id: str
    result: PredictionResult

class BatchResponse(BaseModel):
    predictions: List[BatchResponseItem]

# ========================
# LOAD MODEL ONCE
# ========================

print("--- Khởi động ML Service ---")

# Lấy đường dẫn thư mục gốc (ml_service)
# Giả sử file này ở ml_service/api/api.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

predictor = Predictor(model_path=MODEL_PATH)

app = FastAPI(title="ML Classification Service")

print("--- ML Service sẵn sàng ---")

# ========================
# API ENDPOINTS
# ========================

@app.get("/")
def root():
    return {"status": "ML service OK"}

@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(batch_input: BatchInput):
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for post in batch_input.posts:
            local_image_paths = []

            for url in post.image_urls:
                try:
                    response = requests.get(url, timeout=10, headers=HEADERS)
                    response.raise_for_status()

                    # Code mới (Đã sửa)
                    parsed_url = urlparse(url)  # Tách URL ra
                    clean_path = parsed_url.path  # Chỉ lấy phần đường dẫn, bỏ qua phần ?query...
                    ext = os.path.splitext(clean_path)[1] or ".jpg"  # Lấy đuôi từ đường dẫn sạch

                    filename = f"{uuid.uuid4()}{ext}"
                    save_path = os.path.join(temp_dir, filename)

                    with open(save_path, "wb") as f:
                        f.write(response.content)

                    local_image_paths.append(save_path)


                except requests.exceptions.RequestException as e:
                    print("Lỗi tải ảnh:", url, e)

            # Sửa lỗi cảnh báo `weights_only` của PyTorch
            prediction = predictor.predict(
                text_content=post.text_content,
                image_paths=local_image_paths,
                threshold=0.5
            )

            results.append(BatchResponseItem(
                post_id=post.post_id,
                result=prediction
            ))

    return BatchResponse(predictions=results)

if __name__ == "__main__":
    print("--- CHẠY TRỰC TIẾP (Bằng cách bấm Play ▶️) ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)