import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text, types
from dotenv import load_dotenv
import sys
import urllib  # Cần cho chuỗi kết nối
import datetime  # Cần cho watermark


# --- [BƯỚC 1: KHỞI TẠO] ---

def load_config():
    """Tải cấu hình từ file .env"""
    # load_dotenv() sẽ tự động tìm file .env ở thư mục gốc
    load_dotenv()

    config = {
        "app_conn_str": os.getenv("APP_SQL_SERVER_CONN_STR"),
        "dwh_conn_str": os.getenv("DWH_SQL_SERVER_CONN_STR"),
        "ml_api_url": os.getenv("ML_API_URL")
    }

    if not all(config.values()):
        print("LỖI: Một hoặc nhiều biến môi trường (.env) bị thiếu.")
        print("Vui lòng kiểm tra file .env của bạn.")
        sys.exit(1)

    print("Tải cấu hình từ .env thành công.")
    return config


def create_db_engine(conn_str, db_name):
    """Tạo kết nối (engine) đến CSDL (Đã sửa lỗi parse URL)"""
    try:
        # Mã hóa chuỗi kết nối để nó an toàn khi đặt trong URL
        quoted_conn_str = urllib.parse.quote_plus(conn_str)

        # Định dạng URL "khoa học" mà SQLAlchemy hiểu
        engine_url = f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}"

        engine = create_engine(engine_url)

        # Thử kết nối
        with engine.connect() as conn:
            print(f"Kết nối CSDL {db_name} thành công!")
        return engine

    except Exception as e:
        print(f"LỖI: Không thể kết nối CSDL {db_name}.")
        print(f"Chi tiết lỗi: {e}")
        print("Gợi ý: Bạn đã cài 'ODBC Driver 17 for SQL Server' chưa?")
        sys.exit(1)


# --- [BƯỚC 2: (E) EXTRACT - TRÍCH XUẤT] ---

def get_watermark(dwh_engine):
    """Lấy 'dấu trang' thời gian từ DWH."""
    # Giá trị mặc định nếu bảng không tồn tại hoặc rỗng
    last_timestamp = datetime.datetime(1970, 1, 1)

    check_table_sql = "IF OBJECT_ID('etl_watermarks', 'U') IS NULL SELECT 0 ELSE SELECT 1"

    try:
        with dwh_engine.connect() as conn:
            table_exists = conn.execute(text(check_table_sql)).scalar()

            if table_exists:
                query = "SELECT last_processed_timestamp FROM etl_watermarks WHERE table_name = 'posts'"
                result = conn.execute(text(query)).scalar()
                if result:
                    last_timestamp = result
            else:
                # Nếu bảng không tồn tại, tạo nó
                print("CẢNH BÁO: Bảng 'etl_watermarks' không tồn tại. Đang tạo...")
                create_table_sql = """
                    CREATE TABLE etl_watermarks (
                        table_name VARCHAR(100) PRIMARY KEY,
                        last_processed_timestamp DATETIME
                    );
                    INSERT INTO etl_watermarks (table_name, last_processed_timestamp) 
                    VALUES ('posts', '1970-01-01');
                """
                conn.execute(text(create_table_sql))
                print("Đã tạo bảng 'etl_watermarks'.")

    except Exception as e:
        print(f"LỖI khi lấy watermark: {e}. Sẽ chạy lại từ đầu ('1970-01-01').")

    print(f"  -> Dấu trang (Watermark) hiện tại: create_at > '{last_timestamp}'")
    return last_timestamp


def extract_data(app_engine, dwh_engine, last_timestamp):
    """Trích xuất dữ liệu mới từ App DB DỰA TRÊN DẤU TRANG THỜI GIAN."""
    print("Bắt đầu (E) Extract...")

    # Chuyển đổi datetime sang string mà SQL Server hiểu
    timestamp_str = last_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # 1. Query Posts và Media
    query_posts = f"""
        SELECT 
            p.post_id, p.content, p.create_at AS post_create_at,
            (SELECT STRING_AGG(m.media_url, ',') 
             FROM post_media m 
             WHERE m.post_id = p.post_id) AS media_urls
        FROM 
            posts p
        WHERE 
            p.create_at > '{timestamp_str}' -- Dùng dấu trang THỜI GIAN
        ORDER BY 
            p.create_at ASC -- Rất quan trọng
    """

    # 2. Query Interactions (Đã sửa lỗi `post_shares.create_at`)
    query_interactions = f"""
        SELECT post_id, account_id, 'like' AS interaction_type, 1 AS interaction_weight, create_at 
        FROM likes WHERE create_at > '{timestamp_str}'
        UNION ALL
        SELECT post_id, account_id, 'comment' AS interaction_type, 2 AS interaction_weight, create_at 
        FROM post_comments WHERE create_at > '{timestamp_str}'
        UNION ALL
        SELECT post_id, account_id, 'share' AS interaction_type, 3 AS interaction_weight, create_at 
        FROM post_shares WHERE create_at > '{timestamp_str}'
    """

    try:
        df_posts = pd.read_sql(query_posts, app_engine)
        df_interactions = pd.read_sql(query_interactions, app_engine)

        # 3. TÍNH DẤU TRANG MỚI (MAX TIME)
        new_watermark = None
        if not df_posts.empty:
            new_watermark = df_posts['post_create_at'].max()

        if pd.isna(new_watermark):
            new_watermark = last_timestamp  # Giữ nguyên dấu trang cũ

        print(f"  -> Trích xuất: {len(df_posts)} posts, {len(df_interactions)} tương tác.")
        print(f"  -> Dấu trang (Watermark) mới sẽ là: '{new_watermark}'")

        return df_posts, df_interactions, new_watermark

    except Exception as e:
        print(f"LỖI (E) Extract: {e}")
        return pd.DataFrame(), pd.DataFrame(), last_timestamp


# --- [BƯỚC 3: (T) TRANSFORM - BIẾN ĐỔI] ---

def transform_enrich(df_posts, ml_api_url):
    """Gọi ML Service để làm giàu chủ đề (category)."""
    if df_posts.empty:
        print("Không có post để (T) Transform.")
        # Trả về 3 DataFrame rỗng
        return df_posts, pd.DataFrame(), pd.DataFrame()

    print("Bắt đầu (T) Transform...")

    # 1. Chuẩn bị payload cho ML API
    batch_input = {"posts": []}
    for _, row in df_posts.iterrows():
        # Đảm bảo post_id là string (nếu nó là GUID/UUID)
        post_id_str = str(row['post_id'])
        urls = row['media_urls'].split(',') if pd.notna(row['media_urls']) else []
        batch_input["posts"].append({
            "post_id": post_id_str,
            "text_content": row['content'] or "",
            "image_urls": urls
        })

    # 2. Gọi ML API
    print(f"  -> Đang gọi ML API ({ml_api_url}) cho {len(df_posts)} posts...")
    try:
        response = requests.post(ml_api_url, json=batch_input, timeout=300)
        response.raise_for_status()
        ml_results = response.json()['predictions']

        # 3. Xử lý kết quả (Hỗ trợ đa nhãn)
        bridge_data = []  # Cho bridge_post_categories
        all_categories = set()  # Dùng set để tránh trùng lặp
        post_to_cats = {}  # Dict để map post_id -> [labels]

        for item in ml_results:
            # post_id từ API trả về (string)
            post_id_from_api = item['post_id']

            labels = item['result']['predicted_labels']
            if not labels:
                labels = ['Unknown']

            # Dùng post_id_from_api (string) làm key
            post_to_cats[post_id_from_api] = labels

            for label in labels:
                all_categories.add(label)
                # Dùng post_id_from_api (string)
                bridge_data.append({'post_id': post_id_from_api, 'category_name': label})

            # Sửa luôn ở đây: df_posts['post_id'] có thể là object (GUID),
            # chúng ta cần .astype(str) để nó khớp với key của dict
        df_posts['ml_categories'] = df_posts['post_id'].astype(str).map(post_to_cats)

        df_bridge = pd.DataFrame(bridge_data)
        df_categories = pd.DataFrame(list(all_categories), columns=['category_name'])

        print(f"  -> (T) Transform hoàn tất. Tìm thấy {len(df_categories)} chủ đề.")
        return df_posts, df_categories, df_bridge

    except requests.exceptions.RequestException as e:
        print(f"LỖI (T) Transform: Không thể gọi ML API. {e}")
        # Trả về 3 giá trị rỗng (đã sửa lỗi)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# --- [BƯỚC 4: (L) LOAD - TẢI VÀO DWH] ---

def load_to_dwh(engine, df_posts, df_interactions, df_categories, df_bridge, new_watermark):
    """Tải các DataFrame đã xử lý vào DWH VÀ LƯU DẤU TRANG THỜI GIAN."""
    print("Bắt đầu (L) Load...")

    # Biến cờ để kiểm tra xem có nên cập nhật watermark không
    load_successful = True

    with engine.begin() as conn:  # Mở 1 transaction
        try:
            # --- 1. Tải Dim Categories ---
            existing_cats_df = pd.read_sql("SELECT category_name FROM dim_categories", conn)
            existing_cats = set(existing_cats_df['category_name'])
            new_cats_df = df_categories[~df_categories['category_name'].isin(existing_cats)]

            if not new_cats_df.empty:
                # Định nghĩa kiểu dữ liệu là NVARCHAR
                cat_dtype = {'category_name': types.NVARCHAR(length=100)}

                new_cats_df.to_sql('dim_categories',
                                   conn,
                                   if_exists='append',
                                   index=False,
                                   dtype=cat_dtype)  # <-- Thêm dtype
                print(f"  -> (L) Đã tải {len(new_cats_df)} chủ đề mới vào dim_categories.")

            # Tải lại toàn bộ map (category_name -> category_key)
            all_cats_map_df = pd.read_sql("SELECT category_key, category_name FROM dim_categories", conn)
            cat_name_to_key = dict(zip(all_cats_map_df['category_name'], all_cats_map_df['category_key']))

            # --- 2. Tải Dim Posts ---
            if not df_posts.empty:
                posts_to_load = df_posts[['post_id', 'content', 'post_create_at']]
                posts_dtype = {
                    'content': types.NVARCHAR()  # NVARCHAR() = NVARCHAR(MAX)
                }
                posts_to_load.to_sql('dim_posts', conn, if_exists='append', index=False, dtype=posts_dtype)
                print(f"  -> (L) Đã tải {len(posts_to_load)} posts mới vào dim_posts.")

            # Tải lại toàn bộ map (post_id -> post_key)
            all_posts_map_df = pd.read_sql("SELECT post_key, post_id FROM dim_posts", conn, index_col='post_id')
            post_id_to_key = all_posts_map_df['post_key'].to_dict()

            # --- 3. Tải Bridge Table ---
            if not df_bridge.empty:
                df_bridge['post_key'] = df_bridge['post_id'].map(post_id_to_key)
                df_bridge['category_key'] = df_bridge['category_name'].map(cat_name_to_key)

                bridge_to_load = df_bridge[['post_key', 'category_key']].dropna().drop_duplicates()
                bridge_to_load.to_sql('bridge_post_categories', conn, if_exists='append', index=False)
                print(f"  -> (L) Đã tải {len(bridge_to_load)} liên kết vào bridge_post_categories.")

            # --- 4. Tải Dim Users ---
            if not df_interactions.empty:
                all_account_ids = df_interactions['account_id'].dropna().unique()
                existing_users_df = pd.read_sql("SELECT account_id FROM dim_users", conn)
                existing_users = set(existing_users_df['account_id'])

                new_users_list = [acc_id for acc_id in all_account_ids if acc_id not in existing_users]
                if new_users_list:
                    new_users_df = pd.DataFrame(new_users_list, columns=['account_id'])
                    new_users_df.to_sql('dim_users', conn, if_exists='append', index=False)
                    print(f"  -> (L) Đã tải {len(new_users_df)} users mới vào dim_users.")

                # Tải lại toàn bộ map (account_id -> user_key)
                all_users_map_df = pd.read_sql("SELECT user_key, account_id FROM dim_users", conn,
                                               index_col='account_id')
                user_id_to_key = all_users_map_df['user_key'].to_dict()

                # --- 5. Tải Fact Interactions ---
                df_interactions['user_key'] = df_interactions['account_id'].map(user_id_to_key)
                df_interactions['post_key'] = df_interactions['post_id'].map(post_id_to_key)

                fact_to_load = df_interactions[
                    ['user_key', 'post_key', 'interaction_type', 'interaction_weight', 'create_at']]
                fact_to_load = fact_to_load.rename(columns={'create_at': 'interaction_create_at'})

                fact_to_load.to_sql('fact_post_interactions', conn, if_exists='append', index=False)
                print(f"  -> (L) Đã tải {len(fact_to_load)} tương tác mới vào fact_post_interactions.")

            # --- 6. CẬP NHẬT DẤU TRANG (WATERMARK) ---
            # Chỉ cập nhật nếu tất cả các bước trên thành công
            if new_watermark and new_watermark > get_watermark(engine):  # Đảm bảo là mốc mới
                timestamp_str = new_watermark.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                conn.execute(
                    text(
                        f"UPDATE etl_watermarks SET last_processed_timestamp = '{timestamp_str}' WHERE table_name = 'posts'")
                )
                print(f"  -> (L) Đã cập nhật watermark thành: '{timestamp_str}'")

            print("Hoàn tất (L) Load.")

        except Exception as e:
            load_successful = False  # Gắn cờ lỗi
            print(f"LỖI (L) Load: {e}")
            conn.rollback()  # Hủy bỏ transaction nếu có lỗi
            print("Đã Rollback transaction.")

    return load_successful  # Trả về trạng thái thành công


# --- [HÀM CHÍNH] ---
def main():
    print("===== BẮT ĐẦU ETL JOB =====")

    config = load_config()
    app_engine = create_db_engine(config['app_conn_str'], "App DB")
    dwh_engine = create_db_engine(config['dwh_conn_str'], "DWH")

    # Lấy watermark trước
    last_timestamp = get_watermark(dwh_engine)

    # 3. (E) Trích xuất
    df_posts, df_interactions, new_watermark = extract_data(app_engine, dwh_engine, last_timestamp)

    # Kiểm tra xem có dữ liệu mới không
    if df_posts.empty and df_interactions.empty:
        print("Không có dữ liệu mới. Kết thúc.")
        return

    # 4. (T) Biến đổi (Chỉ xử lý posts)
    df_posts, df_categories, df_bridge = transform_enrich(
        df_posts, config['ml_api_url']
    )

    # 5. (L) Tải (Tải cả posts và interactions)
    load_successful = load_to_dwh(dwh_engine, df_posts, df_interactions, df_categories, df_bridge, new_watermark)

    if load_successful:
        print("===== KẾT THÚC ETL JOB (THÀNH CÔNG) =====")
    else:
        print("===== KẾT THÚC ETL JOB (THẤT BẠI) =====")


if __name__ == "__main__":
    main()