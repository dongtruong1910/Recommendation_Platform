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

        # 1. Tìm thời gian mới nhất trong Posts
        max_post_time = last_timestamp
        if not df_posts.empty:
            max_post_time = df_posts['post_create_at'].max()

        # 2. Tìm thời gian mới nhất trong Interactions
        max_inter_time = last_timestamp
        if not df_interactions.empty:
            max_inter_time = df_interactions['create_at'].max()

        # 3. Dấu trang mới là cái nào LỚN HƠN (Mới hơn)
        # (Dùng hàm max của Python để so sánh 2 datetime)
        new_watermark = max(max_post_time, max_inter_time)

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

def load_to_dwh(engine, df_posts, df_interactions, df_categories, df_bridge, new_watermark, old_watermark):
    """Tải dữ liệu vào DWH và dùng kỹ thuật UPSERT để lưu Watermark."""
    print("Bắt đầu (L) Load...")

    load_successful = True

    # Mở Transaction: Được ăn cả, ngã về không
    with engine.begin() as conn:
        try:
            # --- [PHẦN 1: LOAD DỮ LIỆU CÁC BẢNG (Giữ nguyên code cũ của bạn)] ---

            # 1. Dim Categories
            if not df_categories.empty:
                existing_cats = pd.read_sql("SELECT category_name FROM dim_categories", conn)
                existing_set = set(existing_cats['category_name'])
                new_cats = df_categories[~df_categories['category_name'].isin(existing_set)]
                if not new_cats.empty:
                    # Lưu ý: Cần khai báo dtype cho cột NVARCHAR nếu có dấu
                    new_cats.to_sql('dim_categories', conn, if_exists='append', index=False,
                                    dtype={'category_name': types.NVARCHAR(100)})

            # Load Map Categories
            cat_map = pd.read_sql("SELECT category_key, category_name FROM dim_categories", conn)
            cat_dict = dict(zip(cat_map['category_name'], cat_map['category_key']))

            # 2. Dim Posts
            if not df_posts.empty:
                posts_cut = df_posts[['post_id', 'content', 'post_create_at']].copy()
                posts_cut.to_sql('dim_posts', conn, if_exists='append', index=False,
                                 dtype={'content': types.NVARCHAR()})  # NVARCHAR(MAX)

            # Load Map Posts
            post_map = pd.read_sql("SELECT post_key, post_id FROM dim_posts", conn)
            # Chuyển post_id sang string để map cho chuẩn
            post_dict = dict(zip(post_map['post_id'].astype(str), post_map['post_key']))

            # 3. Bridge Table
            if not df_bridge.empty:
                df_bridge['post_key'] = df_bridge['post_id'].astype(str).map(post_dict)
                df_bridge['category_key'] = df_bridge['category_name'].map(cat_dict)
                # Loại bỏ dòng nào không map được (NaN)
                df_final_bridge = df_bridge[['post_key', 'category_key']].dropna().drop_duplicates()
                df_final_bridge.to_sql('bridge_post_categories', conn, if_exists='append', index=False)

            # 4. Dim Users & Fact Interactions
            if not df_interactions.empty:
                # Tải User mới
                current_users = pd.read_sql("SELECT account_id FROM dim_users", conn)
                user_set = set(current_users['account_id'])
                # Lọc user chưa có trong DB
                incoming_users = df_interactions['account_id'].dropna().unique()
                new_users = [u for u in incoming_users if u not in user_set]

                if new_users:
                    pd.DataFrame({'account_id': new_users}).to_sql('dim_users', conn, if_exists='append', index=False)

                # Load Map Users
                user_map = pd.read_sql("SELECT user_key, account_id FROM dim_users", conn)
                user_dict = dict(zip(user_map['account_id'], user_map['user_key']))

                # Map Fact
                df_interactions['user_key'] = df_interactions['account_id'].map(user_dict)
                df_interactions['post_key'] = df_interactions['post_id'].astype(str).map(post_dict)

                fact_table = df_interactions[
                    ['user_key', 'post_key', 'interaction_type', 'interaction_weight', 'create_at']].copy()
                fact_table.rename(columns={'create_at': 'interaction_create_at'}, inplace=True)
                # Chỉ lấy dòng đủ key
                fact_table = fact_table.dropna()
                fact_table.to_sql('fact_post_interactions', conn, if_exists='append', index=False)

            # --- [PHẦN 2: CẬP NHẬT WATERMARK (SỬA LỖI)] ---
            # Chỉ update nếu có dữ liệu mới hơn
            if new_watermark and new_watermark > old_watermark:
                # Format time chuẩn SQL
                ts_str = new_watermark.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                # Câu lệnh UPSERT: Update trước, nếu không thấy dòng nào thì Insert
                # Ta dùng key là 'global_cursor' (hoặc 'posts' tùy bạn) để đại diện cho cả hệ thống
                sql_upsert = f"""
                    UPDATE etl_watermarks 
                    SET last_processed_timestamp = '{ts_str}' 
                    WHERE table_name = 'posts';

                    IF @@ROWCOUNT = 0
                    BEGIN
                        INSERT INTO etl_watermarks (table_name, last_processed_timestamp)
                        VALUES ('posts', '{ts_str}');
                    END
                """
                conn.execute(text(sql_upsert))
                print(f"  -> (Watermark) Đã lưu thành công mốc thời gian: {ts_str}")

            print("Hoàn tất (L) Load.")

        except Exception as e:
            load_successful = False
            print(f"❌ LỖI (L) Load: {e}")
            # Tự động Rollback, không lưu bất cứ thứ gì (cả data lẫn watermark) để đảm bảo nhất quán
            raise e

    return load_successful

# --- [HÀM CHÍNH] ---
def main():
    print("===== BẮT ĐẦU ETL JOB =====")

    config = load_config()
    app_engine = create_db_engine(config['app_conn_str'], "App DB")
    dwh_engine = create_db_engine(config['dwh_conn_str'], "DWH")

    # 1. Lấy dấu trang cũ
    last_timestamp = get_watermark(dwh_engine)

    # 2. (E) Trích xuất
    df_posts, df_interactions, new_watermark = extract_data(app_engine, dwh_engine, last_timestamp)

    # 3. CHỐT CHẶN 1: Nếu cả 2 đều rỗng -> Dừng
    if df_posts.empty and df_interactions.empty:
        print("-> Không có dữ liệu gì mới. Dừng ETL.")
        print("===== KẾT THÚC ETL JOB =====")
        return

    # 4. (T) Biến đổi (Điều phối thông minh)
    if not df_posts.empty:
        # Có post mới -> Gọi API ML
        df_posts, df_categories, df_bridge = transform_enrich(
            df_posts, config['ml_api_url']
        )
    else:
        print("-> Không có post mới. Bỏ qua bước gọi ML API.")
        # Tạo DataFrame rỗng CÓ TÊN CỘT (để hàm Load không bị lỗi KeyError)
        df_categories = pd.DataFrame(columns=['category_name'])
        df_bridge = pd.DataFrame(columns=['post_id', 'category_name'])

    # 5. (L) Tải
    # Hàm này đã có sẵn "if not empty" nên nó sẽ tự biết cái nào cần tải, cái nào bỏ qua
    load_successful = load_to_dwh(dwh_engine, df_posts, df_interactions, df_categories, df_bridge, new_watermark,last_timestamp)

    # 6. Cập nhật Watermark
    # Chỉ cập nhật nếu có post mới thực sự
    if load_successful and new_watermark and new_watermark > last_timestamp:
        try:
            with dwh_engine.connect() as conn:
                timestamp_str = new_watermark.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                conn.execute(
                    text(
                        f"UPDATE etl_watermarks SET last_processed_timestamp = '{timestamp_str}' WHERE table_name = 'posts'")
                )
                conn.commit()
                print(f"  -> (Watermark) Đã cập nhật watermark thành: '{timestamp_str}'")
        except Exception as e:
            print(f"LỖI CẬP NHẬT WATERMARK: {e}")

    if load_successful:
        print("===== KẾT THÚC ETL JOB (THÀNH CÔNG) =====")
    else:
        print("===== KẾT THÚC ETL JOB (THẤT BẠI) =====")


if __name__ == "__main__":
    main()