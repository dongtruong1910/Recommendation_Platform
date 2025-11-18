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
            print(f"✅ Kết nối DWH (Job: build_feed) thành công!")
        return engine
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối DWH. Chi tiết: {e}")
        sys.exit(1)


def create_redis_client(host, port):
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
        print("✅ Kết nối Redis (Job: build_feed) thành công!")
        return r
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối Redis. Chi tiết: {e}")
        sys.exit(1)


# --- [BƯỚC 2: TÍNH TOÁN (Đọc DWH)] ---
def calculate_scores_from_dwh(dwh_engine):
    print("\n--- Bắt đầu tính toán 'Điểm chất lượng Toàn cầu' từ DWH... ---")
    sql_query = """
        WITH PostPopularity AS (
            SELECT post_key, SUM(interaction_weight) AS total_pop_score
            FROM fact_post_interactions
            WHERE interaction_create_at >= DATEADD(day, -3, GETDATE())
            GROUP BY post_key
        ), PostScores AS (
            SELECT 
                p.post_key, p.post_id, c.category_name,
                EXP(-0.01 * DATEDIFF(hour, p.post_create_at, GETDATE())) AS time_score,
                ISNULL(pop.total_pop_score, 0) AS pop_score
            FROM dim_posts AS p
            JOIN bridge_post_categories AS b ON p.post_key = b.post_key
            JOIN dim_categories AS c ON b.category_key = c.category_key
            LEFT JOIN PostPopularity AS pop ON p.post_key = pop.post_key
            WHERE p.post_create_at >= DATEADD(month, -3, GETDATE())
        )
        SELECT post_key, post_id, category_name,
            (0.7 * time_score) + (0.3 * pop_score) AS global_quality_score
        FROM PostScores;
    """
    try:
        df_scores = pd.read_sql(sql_query, dwh_engine)
        print(f"  -> Đã đọc DWH: {len(df_scores)} cặp (post, category) được chấm điểm.")
        return df_scores
    except Exception as e:
        print(f"❌ LỖI SQL: Không thể tính điểm bài viết. Chi tiết: {e}")
        return pd.DataFrame()


# --- [BƯỚC 3: LẤY DỮ LIỆU TỪ REDIS] ---
def get_user_profiles_from_redis(redis_client):
    print("\n--- Bắt đầu lấy 'Hồ sơ User' từ Redis... ---")
    user_profiles = {}
    profile_keys = redis_client.keys("profile:*")
    if not profile_keys:
        print("  -> ⚠️  CẢNH BÁO: Không tìm thấy hồ sơ user nào trong Redis.")
        return user_profiles

    for key in profile_keys:
        user_key = key.split(":")[1]
        profile_data = redis_client.hgetall(key)
        user_profiles[user_key] = {cat: float(score) for cat, score in profile_data.items()}

    print(f"  -> Lấy xong hồ sơ cho {len(user_profiles)} users.")
    return user_profiles


# --- [BƯỚC 4: TẢI VÀO REDIS - PHIÊN BẢN CUỐI CÙNG] ---
def load_and_fan_out(redis_client, df_scores, user_profiles):
    """
    Tạo các map tra cứu đáng tin cậy, sau đó thực hiện fan-out.
    """
    if df_scores.empty:
        print("⚠️  CẢNH BÁO: Không có điểm bài viết nào để xử lý. Dừng lại.")
        return

    print("\n--- Bắt đầu quy trình Nạp và Fan-out (Phiên bản Cuối cùng)... ---")
    MAX_INBOX_SIZE = 1000
    pipe = redis_client.pipeline()

    # 1. Tạo/Làm mới các Map và Bucket chung một cách an toàn
    print("  -> Chuẩn bị tạo các Map và Bucket chung...")
    pipe.delete("map:post_key_to_guid", "map:post_to_categories", "bucket:GlobalPopular")
    print("  -> Đã xóa các map/bucket cũ do job này quản lý.")

    # Tạo map: post_key -> guid
    df_unique_posts = df_scores[['post_key', 'post_id']].drop_duplicates()
    key_to_guid_map = {str(r['post_key']): str(r['post_id']) for _, r in df_unique_posts.iterrows()}
    if key_to_guid_map:
        pipe.hset("map:post_key_to_guid", mapping=key_to_guid_map)

    # Tạo map: post_key -> "cat1,cat2"
    post_to_cats_map = df_scores.groupby('post_key')['category_name'].apply(lambda x: ','.join(set(x))).to_dict()
    post_to_cats_map_str_keys = {str(k): v for k, v in post_to_cats_map.items()}
    if post_to_cats_map_str_keys:
        pipe.hset("map:post_to_categories", mapping=post_to_cats_map_str_keys)
        print("  -> ✅ Map 'map:post_to_categories' đã được chuẩn bị.")

    # Tạo bucket GlobalPopular
    df_global_pop = df_scores.groupby('post_key')['global_quality_score'].max().nlargest(500)
    if not df_global_pop.empty:
        pipe.zadd("bucket:GlobalPopular", mapping=df_global_pop.to_dict())

    pipe.execute()
    print("  -> ✅ Đã tạo xong các Map và Bucket chung.")

    # (Phần Fan-out và Capping phía sau giữ nguyên, không thay đổi)
    category_to_users = {}
    for user_key, profile in user_profiles.items():
        for category, affinity in profile.items():
            if category not in category_to_users:
                category_to_users[category] = []
            category_to_users[category].append({'key': user_key, 'affinity': affinity})

    print("  -> Bắt đầu quá trình Fan-out thông minh...")
    for _, post_row in df_scores.iterrows():
        post_key = str(post_row['post_key'])
        category = post_row['category_name']
        global_score = float(post_row['global_quality_score'])

        if category in category_to_users:
            for user_info in category_to_users[category]:
                user_key = user_info['key']
                if not redis_client.sismember(f"seen:{user_key}", post_key):
                    personalized_score = global_score * user_info['affinity']
                    redis_client.zadd(f"feed:{user_key}", {post_key: personalized_score})
    print("  -> ✅ Hoàn thành việc đẩy bài mới vào các feed inbox.")

    print("  -> Bắt đầu giới hạn kích thước các feed inbox...")
    final_pipe = redis_client.pipeline()
    for user_key in user_profiles.keys():
        final_pipe.zremrangebyrank(f"feed:{user_key}", 0, -MAX_INBOX_SIZE - 1)
    final_pipe.execute()
    print("  -> ✅ Hoàn thành giới hạn kích thước inbox.")


# --- [HÀM CHÍNH ĐỂ CHẠY FILE] ---
def main():
    print("===== BẮT ĐẦU JOB: BUILD FEEDS (Kiến trúc B) =====")
    config = load_config()
    dwh_engine = create_dwh_engine(config['dwh_conn_str'])
    redis_client = create_redis_client(config['redis_host'], config['redis_port'])

    df_scores = calculate_scores_from_dwh(dwh_engine)
    user_profiles = get_user_profiles_from_redis(redis_client)
    load_and_fan_out(redis_client, df_scores, user_profiles)

    print("===== KẾT THÚC JOB: BUILD FEEDS (Kiến trúc B) =====")


if __name__ == "__main__":
    main()