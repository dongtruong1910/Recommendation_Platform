# Dockerfile (Phiên bản cập nhật)

# 1. Chọn ảnh nền (base image)
FROM python:3.10-slim

# 2. Đặt thư mục làm việc bên trong container
WORKDIR /app

# 3. Sao chép CHỈ file requirements và cài đặt
COPY requirements.txt .
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# 4. Sao chép TOÀN BỘ code của project vào /app
COPY . .

# 5. (Quan trọng) Thêm thư mục gốc vào PYTHONPATH
# Điều này cho phép `api/api.py` có thể `import src`
ENV PYTHONPATH="/app"

# 6. Chỉ định cổng mà API sẽ chạy
EXPOSE 8000

# 7. Lệnh để chạy service API (ĐÃ THAY ĐỔI)
# Trỏ vào module 'api.api' và đối tượng 'app'
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]