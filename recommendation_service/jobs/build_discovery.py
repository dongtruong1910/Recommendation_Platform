import os
import sys
from dotenv import load_dotenv
import redis

# --- [BƯỚC 1: KHỞI TẠO & KẾT NỐI] ---
# (Phần này giữ nguyên, không thay đổi)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_config():
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
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
        print("✅ Kết nối Redis (Job: build_discovery) thành công!")
        return r
    except Exception as e:
        print(f"❌ LỖI: Không thể kết nối Redis. Chi tiết: {e}")
        sys.exit(1)


# --- [BƯỚC 2: TÍNH TOÁN VÀ TẢI - PHIÊN BẢN SỬA LỖI CUỐI CÙNG] ---
def build_discovery_feeds_final(redis_client):
    """
    Tạo inbox khám phá bằng cách đọc map đã được chuẩn bị sẵn,
    đảm bảo logic lọc chính xác 100%.
    """
    print("\n--- Bắt đầu xây dựng 'Discovery Feeds' (Phiên bản Cuối cùng)... ---")
    MAX_DISCOVERY_ITEMS = 100
    KNOWN_CATEGORY_THRESHOLD = 0.05

    try:
        # 1. Lấy tất cả dữ liệu nguồn cần thiết từ Redis
        print("  -> Đang lấy dữ liệu nguồn từ Redis...")

        # === [THAY ĐỔI CỐT LÕI] ===
        # Đọc trực tiếp map đã được job 'build_feed' tạo ra
        # Đây là nguồn dữ liệu đáng tin cậy
        all_post_categories_raw = redis_client.hgetall("map:post_to_categories")
        if not all_post_categories_raw:
            print("⚠️  CẢNH BÁO: Không tìm thấy 'map:post_to_categories'. Job 'build_feed' cần chạy trước. Dừng lại.")
            return

        # Chuyển từ string "cat1,cat2" thành set {'cat1', 'cat2'}
        all_post_categories = {post: set(cats.split(',')) for post, cats in all_post_categories_raw.items()}
        print(f"  -> ✅ Đã đọc thành công bản đồ cho {len(all_post_categories)} bài viết.")
        # === [HẾT THAY ĐỔI CỐT LÕI] ===

        global_popular_posts_with_scores = redis_client.zrevrange("bucket:GlobalPopular", 0, -1, withscores=True)
        user_keys = [key.split(":")[1] for key in redis_client.keys("profile:*")]

        if not global_popular_posts_with_scores or not user_keys:
            print("⚠️  CẢNH BÁO: Thiếu dữ liệu nguồn (`GlobalPopular` hoặc `profile`). Dừng lại.")
            return

        print(f"  -> Nguồn: {len(global_popular_posts_with_scores)} bài từ GlobalPopular.")
        print(f"  -> Mục tiêu: {len(user_keys)} users.")

    except Exception as e:
        print(f"❌ LỖI khi đọc dữ liệu nguồn từ Redis. Chi tiết: {e}")
        return

    # 2. Xây dựng inbox khám phá cho từng user (LOGIC LỌC GIỮ NGUYÊN)
    print("\n--- Bắt đầu lặp qua từng user để tạo Discovery Inbox... ---")
    users_processed = 0
    for user_key in user_keys:
        try:
            profile_data = redis_client.hgetall(f"profile:{user_key}")
            known_categories = {cat for cat, score in profile_data.items() if float(score) >= KNOWN_CATEGORY_THRESHOLD}
            seen_posts = redis_client.smembers(f"seen:{user_key}")
            print(f"  - User {user_key}: có {len(known_categories)} chủ đề đã biết, đã xem {len(seen_posts)} bài.")

            discovery_candidates = {}
            for post, score in global_popular_posts_with_scores:
                if post in seen_posts:
                    continue

                # Logic lọc giờ sẽ chạy đúng vì `all_post_categories` đã đầy đủ
                post_cats = all_post_categories.get(post, set())
                if not post_cats.isdisjoint(known_categories):
                    continue
                discovery_candidates[post] = score

            final_mapping = dict(list(discovery_candidates.items())[:MAX_DISCOVERY_ITEMS])

            key = f"discovery_feed:{user_key}"
            pipe = redis_client.pipeline()
            pipe.delete(key)
            if final_mapping:
                pipe.zadd(key, mapping=final_mapping)
            pipe.execute()

            print(f"    -> ✅ Đã tạo inbox khám phá với {len(final_mapping)} bài.")
            users_processed += 1
        except Exception as e:
            print(f"  -> ❌ Lỗi khi xử lý cho user_key {user_key}. {e}")

    print(f"\n✅ Hoàn thành xây dựng Discovery Feeds cho {users_processed}/{len(user_keys)} users.")


# --- [HÀM CHÍNH ĐỂ CHẠY FILE] ---
def main():
    print("===== BẮT ĐẦU JOB: BUILD DISCOVERY FEEDS (Sửa lỗi) =====")
    config = load_config()
    redis_client = create_redis_client(config['redis_host'], config['redis_port'])
    build_discovery_feeds_final(redis_client)
    print("===== KẾT THÚC JOB: BUILD DISCOVERY FEEDS (Sửa lỗi) =====")


if __name__ == "__main__":
    main()