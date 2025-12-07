# Dockerfile (Dùng chung cho cả API và Worker)

# 1. Chọn ảnh nền (dùng chung version với ETL cho đồng bộ)
FROM python:3.10-slim-bullseye

# 2. Đặt thư mục làm việc
WORKDIR /app

# 3. Cài đặt các gói hệ thống và ODBC Driver (BẮT BUỘC để đọc DWH)
RUN apt-get update && apt-get install -y gnupg curl \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements trước để tận dụng Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code (api/, jobs/, rec_scheduler.py...)
COPY . .

# 6. Thêm PYTHONPATH để Python tìm thấy các module
ENV PYTHONPATH="/app"

# 7. Mở cổng 7001 (Thông báo cho Docker biết)
EXPOSE 7001

# 8. Lệnh chạy mặc định
# (Mặc định là chạy API. Khi chạy Worker, docker-compose sẽ ghi đè lệnh này)
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "7001"]