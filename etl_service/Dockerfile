# 1. Dùng phiên bản Bullseye (Debian 11) để tương thích tốt nhất với MS SQL Driver 17
FROM python:3.10-slim-bullseye

# 2. Đặt thư mục làm việc
WORKDIR /app

# 3. Cài đặt Driver SQL Server (Gộp lệnh để tránh lỗi apt-key not found)
# Lưu ý: Chúng ta cài 'gnupg' và 'curl' TRƯỚC, sau đó mới dùng được apt-key
RUN apt-get update && apt-get install -y gnupg curl \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code
COPY . .

# 6. Thiết lập đường dẫn
ENV PYTHONPATH="/app"

# 7. Mở cổng (Chỉ có tác dụng với API, Worker sẽ bỏ qua)
EXPOSE 8001

# 8. Lệnh chạy mặc định (Sẽ bị ghi đè bởi docker-compose)
CMD ["python", "etl_scheduler.py"]
# (Hoặc CMD ["uvicorn", ...] tùy bạn để mặc định là gì, docker-compose sẽ lo phần này)