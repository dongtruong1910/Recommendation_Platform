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
    """Tải các biến môi trường cần thiết."""
    load_dotenv()
    config = {
        "dwh_conn_str": os.getenv("DWH_SQL_SERVER_CONN_STR"),
        "redis_host": os.getenv("REDIS_HOST"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379))
    }
    if not all([config["dwh_conn_str"], config["redis_host"]]):
        print("LỖI: Thiếu biến môi trường DWH_SQL_SERVER_CONN_STR hoặc REDIS_HOST/REDIS_PORT.")
        sys.exit(1)
    return config


def create_dwh_engine(conn_str):
    """Tạo SQLAlchemy engine để kết nối DWH."""
    try:
        quoted_conn_str = urllib.parse.quote_plus(conn_str)
        engine_url = f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}"
        engine = create_engine(engine_url)
        with engine.connect() as conn:
            print(f"✅ Kết nối DWH (Job: update_profile) thành công!")
        return engine
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối DWH. Chi tiết: {e}")
        sys.exit(1)


def create_redis_client(host, port):
    """Tạo Redis client."""
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
        print("✅ Kết nối Redis (Job: update_profile) thành công!")
        return r
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối Redis. Chi tiết: {e}")
        sys.exit(1)


# --- [BƯỚC 2: TÍNH TOÁN (Đọc DWH)] ---
def calculate_profiles_from_dwh(dwh_engine):
    """Chạy SQL để lấy điểm tương tác thô và bảng tra cứu user."""
    print("\n--- Bắt đầu tính toán 'Hồ sơ User' từ DWH... ---")

    sql_query_scores = """
        SELECT 
            f.user_key, c.category_name, SUM(f.interaction_weight) AS total_score
        FROM fact_post_interactions AS f
        JOIN bridge_post_categories AS b ON f.post_key = b.post_key
        JOIN dim_categories AS c ON b.category_key = c.category_key
        WHERE f.interaction_create_at >= DATEADD(month, -3, GETDATE())
        GROUP BY f.user_key, c.category_name;
    """

    sql_query_user_map = "SELECT user_key, account_id FROM dim_users;"

    try:
        df_scores = pd.read_sql(sql_query_scores, dwh_engine)
        print(f"  -> Đã đọc DWH: {len(df_scores)} cặp (user, category) được chấm điểm.")

        df_user_map = pd.read_sql(sql_query_user_map, dwh_engine)
        print(f"  -> Đã đọc DWH: Bảng tra cứu cho {len(df_user_map)} users.")

        return df_scores, df_user_map
    except Exception as e:
        print(f"❌ LỖI SQL: Không thể tính điểm hồ sơ. Chi tiết: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- [BƯỚC 3: XỬ LÝ (Pandas) VÀ TẢI (Redis)] ---
def process_and_load_profiles(redis_client, df_scores, df_user_map):
    """Chuẩn hóa điểm số thành tỷ lệ % và nạp vào Redis."""
    if df_scores.empty:
        print("⚠️  CẢNH BÁO: Không có điểm hồ sơ nào để xử lý. Dừng lại.")
        return

    print("\n--- Bắt đầu Chuẩn hóa (Normalize) và nạp (load) Profiles & Map vào Redis... ---")
    pipe = redis_client.pipeline()

    # 1. Xóa các key cũ để đảm bảo dữ liệu mới
    print("  -> Chuẩn bị xóa các `profile:*` và `map:account_id_to_user_key` cũ...")
    old_profile_keys = redis_client.keys("profile:*")
    if old_profile_keys:
        pipe.delete(*old_profile_keys)
    pipe.delete("map:account_id_to_user_key")

    # 2. Tải Bảng tra cứu (Lookup Map) account_id -> user_key
    if not df_user_map.empty:
        print("  -> Chuẩn bị Bảng tra cứu (map:account_id_to_user_key)...")
        account_to_user_key_map = {
            str(row['account_id']): str(row['user_key'])
            for _, row in df_user_map.iterrows()
        }
        pipe.hset("map:account_id_to_user_key", mapping=account_to_user_key_map)
    else:
        print("  -> ⚠️ CẢNH BÁO: Không có dữ liệu `df_user_map` để tạo bảng tra cứu.")

    # 3. Tính % (Normalize)
    print("  -> Đang chuẩn hóa điểm số thành tỷ lệ %...")
    df_total_scores = df_scores.groupby('user_key')['total_score'].sum().reset_index()
    df_total_scores = df_total_scores.rename(columns={'total_score': 'user_total_score'})
    df_norm = pd.merge(df_scores, df_total_scores, on='user_key')
    # Tránh chia cho 0 nếu user_total_score = 0
    df_norm = df_norm[df_norm['user_total_score'] > 0]
    df_norm['percentage'] = df_norm['total_score'] / df_norm['user_total_score']

    # 4. Nhóm theo user_key để chuẩn bị ghi vào Redis
    grouped = df_norm.groupby('user_key')
    print(f"  -> Chuẩn bị nạp hồ sơ cho {len(grouped)} users...")
    for user_key, group_df in grouped:
        key = f"profile:{user_key}"
        mapping = {
            row['category_name']: round(float(row['percentage']), 4)  # Làm tròn cho đẹp
            for _, row in group_df.iterrows()
        }
        pipe.hset(key, mapping=mapping)

    # 5. Thực thi pipeline
    try:
        results = pipe.execute()
        print(f"✅ Nạp User Profiles và Map vào Redis hoàn tất. {len(results)} lệnh đã được thực thi.")
    except Exception as e:
        print(f"❌ LỖI khi thực thi pipeline Redis. Chi tiết: {e}")


# --- [HÀM CHÍNH ĐỂ CHẠY FILE] ---
def main():
    """Hàm chính để điều phối toàn bộ job."""
    print("===== BẮT ĐẦU JOB: UPDATE PROFILES =====")
    config = load_config()
    dwh_engine = create_dwh_engine(config['dwh_conn_str'])
    redis_client = create_redis_client(config['redis_host'], config['redis_port'])

    df_scores, df_user_map = calculate_profiles_from_dwh(dwh_engine)
    process_and_load_profiles(redis_client, df_scores, df_user_map)

    print("===== KẾT THÚC JOB: UPDATE PROFILES =====")


if __name__ == "__main__":
    main()