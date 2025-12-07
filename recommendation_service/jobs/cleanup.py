import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib
import redis

# --- [BƯỚC 1: KHỞI TẠO & KẾT NỐI] ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_config():
    load_dotenv()
    config = {
        "dwh_conn_str": os.getenv("DWH_SQL_SERVER_CONN_STR"),
        "redis_host": os.getenv("REDIS_HOST"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379))
    }
    if not all([config["dwh_conn_str"], config["redis_host"]]):
        print("LỖI: Thiếu biến môi trường DWH hoặc REDIS.")
        sys.exit(1)
    return config


def create_dwh_engine(conn_str):
    try:
        quoted_conn_str = urllib.parse.quote_plus(conn_str)
        engine_url = f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}"
        engine = create_engine(engine_url)
        with engine.connect() as conn:
            print(f"✅ Kết nối DWH (Job: cleanup) thành công!")
        return engine
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối DWH. Chi tiết: {e}")
        sys.exit(1)


def create_redis_client(host, port):
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
        print("✅ Kết nối Redis (Job: cleanup) thành công!")
        return r
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối Redis. Chi tiết: {e}")
        sys.exit(1)


# --- [BƯỚC 2: LOGIC DỌN DẸP - PHIÊN BẢN CUỐI CÙNG] ---
def cleanup_old_seen_records_final(dwh_engine, redis_client):
    """
    Dọn dẹp các post_key đã cũ (hơn 3 tháng) ra khỏi tất cả các 'seen' set.
    """
    print("\n--- Bắt đầu Job dọn dẹp 'seen' records cũ (Phiên bản Cuối cùng)... ---")
    EXPIRED_POSTS_KEY = "temp:expired_posts"

    # 1. Lấy danh sách các post_key đã HẾT HẠN (cũ hơn 3 tháng) từ DWH
    sql_query = "SELECT DISTINCT post_key FROM dim_posts WHERE post_create_at < DATEADD(month, -3, GETDATE());"
    try:
        df_expired_posts = pd.read_sql(sql_query, dwh_engine)
        expired_post_keys = {str(pk) for pk in df_expired_posts['post_key']}

        if not expired_post_keys:
            print("  -> ✅ Không tìm thấy bài viết nào cũ hơn 3 tháng. Không có gì để dọn dẹp.")
            return

        print(f"  -> Tìm thấy {len(expired_post_keys)} post_key đã hết hạn trong DWH.")

        # Tải danh sách hết hạn vào một Set tạm thời trong Redis để dùng SINTER
        pipe = redis_client.pipeline()
        pipe.delete(EXPIRED_POSTS_KEY)
        pipe.sadd(EXPIRED_POSTS_KEY, *expired_post_keys)
        pipe.expire(EXPIRED_POSTS_KEY, 60 * 60)  # Tự xóa key tạm sau 1 giờ
        pipe.execute()

    except Exception as e:
        print(f"❌ LỖI: Không thể lấy hoặc tải danh sách post hết hạn. {e}")
        return

    # 2. Lặp qua tất cả các 'seen' set và dọn dẹp
    cursor = '0'
    total_cleaned_count = 0
    while cursor != 0:
        cursor, keys = redis_client.scan(cursor=cursor, match="seen:*", count=100)
        if not keys: continue

        print(f"  -> Quét được {len(keys)} 'seen' sets để kiểm tra...")
        for seen_key in keys:
            try:
                # Tìm những bài đã xem VÀ nằm trong danh sách hết hạn
                # SINTERSTORE hiệu quả hơn: tìm giao và lưu vào key mới, rồi đổi tên
                common_keys = redis_client.sinter(seen_key, EXPIRED_POSTS_KEY)

                if common_keys:
                    print(f"    -> Tìm thấy {len(common_keys)} record hết hạn trong {seen_key}. Đang xóa...")
                    redis_client.srem(seen_key, *common_keys)
                    total_cleaned_count += len(common_keys)
            except Exception as e:
                print(f"    -> ❌ Lỗi khi dọn dẹp key {seen_key}: {e}")

    # Xóa key tạm
    redis_client.delete(EXPIRED_POSTS_KEY)

    if total_cleaned_count > 0:
        print(f"\n✅ Hoàn thành Job dọn dẹp. Tổng cộng đã xóa {total_cleaned_count} record cũ.")
    else:
        print("\n✅ Hoàn thành Job dọn dẹp. Không có record nào cần xóa trong các 'seen' set.")


# --- [HÀM CHÍNH ĐỂ CHẠY FILE] ---
def main():
    print("===== BẮT ĐẦU JOB: CLEANUP (Phiên bản Cuối cùng) =====")
    config = load_config()
    dwh_engine = create_dwh_engine(config['dwh_conn_str'])
    redis_client = create_redis_client(config['redis_host'], config['redis_port'])
    cleanup_old_seen_records_final(dwh_engine, redis_client)
    print("===== KẾT THÚC JOB: CLEANUP (Phiên bản Cuối cùng) =====")


if __name__ == "__main__":
    main()