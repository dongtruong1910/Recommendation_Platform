
import sys
import os
import uvicorn  # Import uvicorn lên đầu

# Lấy đường dẫn thư mục gốc (ml_service)
CURRENT_FILE_PATH = os.path.abspath(__file__)
API_DIR = os.path.dirname(CURRENT_FILE_PATH)
ROOT_DIR = os.path.dirname(API_DIR)

# 1. Bắt Python "chuyển" về thư mục gốc (ml_service)
os.chdir(ROOT_DIR)
print(f"--- Đã đổi thư mục làm việc (CWD) về: {os.getcwd()}")

# 2. Thêm thư mục gốc vào path để tìm 'src'
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import torch
import json
import tempfile
import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any


try:
    from src import config
    from src.exp_v2_cnn.predict_v2 import PredictorV2
except ImportError as e:
    print(f"\n🔥 LỖI IMPORT SRC V2: {e}")
    print("Hãy đảm bảo bạn đã tạo 'src/predict_v2.py'")
    raise

# ========================
# LOAD MODEL V2 ONCE
# ========================

print("--- Khởi động ML Service V2 (CNN) ---")


MODEL_V2_PATH = os.path.join(ROOT_DIR, "models", "best_model_v2_cnn.pth")

if not os.path.exists(MODEL_V2_PATH):
    raise FileNotFoundError(f"Không tìm thấy model V2 tại: {MODEL_V2_PATH}")

# Khởi tạo Predictor V2
predictor_v2 = PredictorV2(model_path=MODEL_V2_PATH)

app = FastAPI(title="ML Service V2 - Playground")

print("--- ML Service V2 sẵn sàng ---")


# ========================
# API ENDPOINTS
# ========================

@app.get("/", response_class=HTMLResponse)
async def get_playground():
    """
    Endpoint này trả về file HTML (giao diện test)
    """
    html_path = os.path.join(API_DIR, "index_v2.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>LỖI: Không tìm thấy file index_v2.html</h1>", status_code=404)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/predict_v2")
async def predict_v2(
        # API này nhận 2 phần: 1 text, 1 file
        text_content: str = Form(...),
        image_file: UploadFile = File(...)
):
    """
    Endpoint này nhận (1 text, 1 ảnh) và trả về dự đoán V2
    """

    # 1. Lưu file ảnh (UploadFile) xuống thư mục tạm
    try:
        # Lấy đuôi file (ví dụ: .jpg, .png)
        ext = os.path.splitext(image_file.filename)[1] or ".jpg"

        # Tạo file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_f:
            # Đọc nội dung file upload và ghi vào file tạm
            await image_file.seek(0)
            content = await image_file.read()
            temp_f.write(content)
            temp_path = temp_f.name  # Lấy đường dẫn file tạm

        print(f"Đã lưu ảnh tạm tại: {temp_path}")

        # 2. Chạy dự đoán (dùng file tạm)
        prediction = predictor_v2.predict(
            text_content=text_content,
            image_path=temp_path,  # Dùng file 1-ảnh
            threshold=0.5
        )

        return prediction

    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. Luôn luôn xóa file tạm sau khi dùng
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Đã xóa ảnh tạm: {temp_path}")



if __name__ == "__main__":
    print("--- CHẠY TRỰC TIẾP (Bằng cách bấm Play ▶️) ---")
    uvicorn.run(app, host="0.0.0.0", port=8001)