# Dockerfile (cho recommendation_service)

# 1. Chọn ảnh nền (dùng chung 3.10-slim)
FROM python:3.10-slim

# 2. Đặt thư mục làm việc
WORKDIR /app

# 3. [RẤT QUAN TRỌNG] Cài đặt ODBC Driver (Giống ETL)
# (Vì các "Job" của service này cần đọc DWH)
RUN apt-get update && apt-get install -y gnupg curl

# Thêm kho lưu trữ (repository) của Microsoft
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Cài đặt driver
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17

# 4. Sao chép CHỈ file requirements và cài đặt
# (Bao gồm fastapi, redis, pyodbc, sqlalchemy...)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Sao chép TOÀN BỘ code (api/, jobs/, .env...)
# (Nhớ tạo .dockerignore để bỏ .venv)
COPY . .

# 6. Thêm PYTHONPATH (cho "khoa học")
ENV PYTHONPATH="/app"

# 7. Mở cổng cho API
EXPOSE 8001

# 8. Lệnh chạy
# Chạy API (Quầy hàng) bằng Uvicorn
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8001"]