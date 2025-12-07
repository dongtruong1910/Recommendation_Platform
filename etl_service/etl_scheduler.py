# etl_scheduler.py
import time
import logging

import schedule

from jobs.run_etl import main as run_etl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - ETL SERVICE - %(message)s')

def run_etl_task():
    logging.info("🚀 BẮT ĐẦU: ETL Sync Job")
    try:
        run_etl()
        logging.info("✅ KẾT THÚC: ETL Sync hoàn tất.")
    except Exception as e:
        logging.error(f"❌ LỖI ETL: {e}")

# ETL nên chạy tần suất cao hơn (ví dụ 10-15 phút/lần) để DWH luôn tươi mới
schedule.every(4).minutes.do(run_etl_task)

print("⏳ ETL Service đã khởi động...")
run_etl_task() # Chạy ngay lần đầu

while True:
    schedule.run_pending()
    time.sleep(60)