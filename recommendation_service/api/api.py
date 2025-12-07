import os
import sys
import uvicorn
import random
import math
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import redis
from typing import List

# --- [BƯỚC 1: KHỞI TẠO & KẾT NỐI] ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_api_config():
    """Tải cấu hình Redis cho API."""
    load_dotenv()
    config = {
        "redis_host": os.getenv("REDIS_HOST"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379))
    }
    if not config["redis_host"]:
        print("LỖI: Thiếu biến môi trường REDIS.")
        sys.exit(1)
    return config


def create_redis_client(host, port):
    """Tạo Redis client và kiểm tra kết nối."""
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
        print("✅ Kết nối Redis (API) thành công!")
        return r
    except Exception as e:
        print(f"❌ LỖI: API không thể kết nối Redis. Chi tiết: {e}")
        sys.exit(1)


app = FastAPI(title="Recommendation Service API")
api_config = load_api_config()
redis_client = create_redis_client(api_config['redis_host'], api_config['redis_port'])
print("--- Recommendation API sẵn sàng phục vụ ---")


# --- [BƯỚC 2: API ENDPOINT - PHIÊN BẢN INBOX] ---
@app.get("/", summary="Check trạng thái API")
def root():
    return {"status": "Recommendation Service API đang hoạt động!"}


@app.get("/feed/{account_id}", response_model=List[str])
async def get_feed(account_id: str, page_size: int = 20):
    """
    API phục vụ feed theo kiến trúc INBOX - siêu tốc.
    """
    print(f"\n--- Yêu cầu Feed cho account_id: {account_id}, page_size: {page_size} ---")

    try:
        # 1. Tra cứu user_key
        user_key = redis_client.hget("map:account_id_to_user_key", account_id)

        # 2. XỬ LÝ LUỒNG USER "NÓNG" (HOT USER)
        if user_key:
            print(f"  -> (User Nóng) Đã tìm thấy user_key: {user_key}.")

            # Phân bổ số lượng
            explore_ratio = 0.1  # 10% khám phá
            explore_count = max(1, math.ceil(page_size * explore_ratio))
            personal_count = page_size - explore_count
            print(f"  -> Phân bổ: {personal_count} bài cá nhân hóa, {explore_count} bài khám phá.")

            feed_key = f"feed:{user_key}"
            discovery_feed_key = f"discovery_feed:{user_key}"
            seen_key = f"seen:{user_key}"

            # Đọc đồng thời 2 inbox bằng pipeline
            read_pipe = redis_client.pipeline()
            read_pipe.zrevrange(feed_key, 0, personal_count - 1)
            read_pipe.zrevrange(discovery_feed_key, 0, explore_count - 1)
            personal_posts, explore_posts = read_pipe.execute()

            print(f"  -> Đã đọc từ inbox: {len(personal_posts)} bài cá nhân hóa, {len(explore_posts)} bài khám phá.")

            # Gộp lại và loại bỏ trùng lặp (dù bây giờ khả năng trùng rất thấp)
            posts_to_show = list(dict.fromkeys(personal_posts + explore_posts))

            #Người dùng đã xem hết post trong 3 tháng gần đây
            if not posts_to_show:
                print("  -> ⚠️ CẢNH BÁO: Cả 2 inbox đều rỗng. Chuyển sang luồng dự phòng.")
            else:
                random.shuffle(posts_to_show)

                print(f"  -> Sẽ hiển thị {len(posts_to_show)} bài DUY NHẤT. Chuẩn bị đánh dấu đã xem...")
                write_pipe = redis_client.pipeline()
                if personal_posts:
                    write_pipe.zrem(feed_key, *personal_posts)
                if explore_posts:
                    write_pipe.zrem(discovery_feed_key, *explore_posts)
                write_pipe.sadd(seen_key, *posts_to_show)
                write_pipe.execute()

                # Dịch sang GUID và trả về
                guid_list = redis_client.hmget("map:post_key_to_guid", posts_to_show)
                return [guid for guid in guid_list if guid]

        # 3. LUỒNG DỰ PHÒNG (USER LẠNH hoặc INBOX RỖNG)
        print("  -> (User Lạnh / Dự phòng) Lấy từ `bucket:GlobalPopular`.")
        fallback_posts = redis_client.zrevrange("bucket:GlobalPopular", 0, page_size - 1)

        if not fallback_posts:
            print("  -> ⚠️ Toàn hệ thống không có bài viết nào.")
            return []

        # Với user lạnh, không cần đánh dấu đã xem để đơn giản hóa
        guid_list = redis_client.hmget("map:post_key_to_guid", fallback_posts)
        return [guid for guid in guid_list if guid]

    except Exception as e:
        import traceback
        print(f"❌ LỖI NGHIÊM TRỌNG trong API: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Lỗi server khi lấy feed.")


# --- [HÀM ĐỂ CHẠY FILE] ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7001, reload=True)