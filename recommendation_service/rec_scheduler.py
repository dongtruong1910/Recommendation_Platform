# rec_scheduler.py (Tên cũ là scheduler.py)
import time
import schedule
import logging
# KHÔNG import ETL ở đây nữa
from jobs.update_profile import main as run_update_profile
from jobs.build_feed import main as run_build_feed
from jobs.build_discovery import main as run_build_discovery
from jobs.cleanup import main as run_cleanup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - REC WORKER - %(message)s')

def job_pipeline_algo():
    """Chỉ chạy thuật toán trên dữ liệu đã có sẵn trong DWH"""
    logging.info("🧠 BẮT ĐẦU: Recommendation Algorithm Pipeline")
    try:
        # Bước 1: Tính Profile
        run_update_profile()
        # Bước 2: Tạo Feed
        run_build_feed()
        # Bước 3: Tạo Discovery
        run_build_discovery()
        logging.info("✅ KẾT THÚC: Pipeline thuật toán hoàn tất.")
    except Exception as e:
        logging.error(f"❌ LỖI ALGO: {e}")

def job_maintenance():
    run_cleanup()

# Thuật toán chạy thưa hơn vì nó tốn CPU
schedule.every(10).minutes.do(job_pipeline_algo)
schedule.every().day.at("03:00").do(job_maintenance)

print("⏳ Recommendation Worker đã khởi động...")
job_pipeline_algo()

while True:
    schedule.run_pending()
    time.sleep(60)