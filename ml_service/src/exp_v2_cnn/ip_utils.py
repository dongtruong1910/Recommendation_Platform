import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from scipy import ndimage


def custom_rgb_to_grayscale(image_rgb):
    """
    Tự triển khai thuật toán RGB -> Grayscale (Luma formula).

    Args:
        image_rgb (np.array): Ảnh RGB (shape [H, W, 3]).

    Returns:
        np.array: Ảnh xám (shape [H, W]).
    """
    # Đảm bảo kiểu dữ liệu là float để nhân, tránh tràn số (overflow)
    img_float = image_rgb.astype(np.float32)

    # Tách 3 kênh
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]

    # Áp dụng công thức Luma
    # Grayscale = 0.299*R + 0.587*G + 0.114*B
    gray = (0.299 * r) + (0.587 * g) + (0.114 * b)

    # Chuyển về kiểu uint8 (0-255)
    return gray.astype(np.uint8)

def otsu_threshold(image_gray):
    """
    Tự triển khai thuật toán Otsu (không dùng cv2.threshold).

    Args:
        image_gray (np.array): Ảnh xám (2D array, 0-255).

    Returns:
        int: Ngưỡng Otsu tốt nhất.
    """
    # 1. Tính toán histogram
    # .ravel() biến ảnh 2D thành mảng 1D
    pixel_counts = np.histogram(image_gray.ravel(), bins=256, range=(0, 256))[0]
    total_pixels = image_gray.size

    # Tính tổng số pixel * cường độ
    current_sum = 0
    all_pixel_sum = np.dot(pixel_counts, np.arange(256))

    current_pixel_count = 0
    max_variance = 0
    best_threshold = 0

    # 2. Lặp qua tất cả 255 ngưỡng có thể
    for t in range(256):
        # Cập nhật số pixel và tổng cường độ cho "lớp" 0 (nền)
        current_pixel_count += pixel_counts[t]
        current_sum += t * pixel_counts[t]

        # Nếu không có pixel, bỏ qua
        if current_pixel_count == 0:
            continue

        # Tính số pixel cho "lớp" 1 (vật thể)
        count_1 = total_pixels - current_pixel_count
        if count_1 == 0:
            break  # Đã đi hết ảnh

        # 3. Tính toán trọng số và trung bình
        weight_0 = current_pixel_count / total_pixels
        weight_1 = count_1 / total_pixels

        mean_0 = current_sum / current_pixel_count
        mean_1 = (all_pixel_sum - current_sum) / count_1

        # 4. Tính phương sai giữa 2 lớp (between-class variance)
        between_class_variance = weight_0 * weight_1 * ((mean_0 - mean_1) ** 2)

        # 5. Tìm phương sai lớn nhất
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            best_threshold = t

    return best_threshold


def custom_erosion(binary_image, kernel):
    """
    Tự triển khai phép Erosion (Co) bằng vòng lặp for.
    Nguyên tắc: Chỉ giữ pixel (1) nếu *tất cả* lân cận (dưới kernel) đều là 1.
    """
    k_height, k_width = kernel.shape
    pad_h = k_height // 2
    pad_w = k_width // 2

    # Pad ảnh để xử lý biên
    padded_image = np.pad(binary_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output_image = np.zeros_like(binary_image)

    # Vòng lặp "trượt" (sliding window)
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            # Lấy vùng lân cận (neighborhood)
            region = padded_image[y: y + k_height, x: x + k_width]

            # Phép nhân logic (AND) giữa kernel và vùng lân cận
            # (Giả sử kernel toàn số 1)
            # Nếu tất cả (all) các pixel trong vùng đều là 1
            if np.all(region == 1):
                output_image[y, x] = 1

    return output_image


def custom_dilation(binary_image, kernel):
    """
    Tự triển khai phép Dilation (Giãn) bằng vòng lặp for.
    Nguyên tắc: Giữ pixel (1) nếu *bất kỳ* lân cận (dưới kernel) là 1.
    """
    k_height, k_width = kernel.shape
    pad_h = k_height // 2
    pad_w = k_width // 2

    # Pad ảnh để xử lý biên
    padded_image = np.pad(binary_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output_image = np.zeros_like(binary_image)

    # Vòng lặp "trượt"
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            region = padded_image[y: y + k_height, x: x + k_width]

            # Phép nhân logic (OR)
            # Nếu có bất kỳ (any) pixel nào trong vùng là 1
            if np.any(region == 1):
                output_image[y, x] = 1

    return output_image

def morph_opening(binary_image, kernel_size=3):
    """
    Thực hiện phép Opening (Erosion -> Dilation)[cite: 122].

    Args:
        binary_image (np.array): Ảnh nhị phân (True/False hoặc 1/0).
        kernel_size (int): Kích thước của kernel (ví dụ: 3x3).

    Returns:
        np.array: Ảnh nhị phân sau khi Opening.
    """
    # Tạo kernel (structuring element)
    kernel = np.ones((kernel_size, kernel_size))

    # # 1. Erosion (Co)
    # eroded = custom_erosion(image_int, kernel)
    # # 2. Dilation (Giãn)
    # opened = custom_dilation(eroded, kernel)
    #
    # return opened.astype(bool)  # Trả về dạng bool cho dataset

    # 1. Erosion (Co)
    eroded = ndimage.binary_erosion(binary_image, structure=kernel)
    # 2. Dilation (Giãn)
    opened = ndimage.binary_dilation(eroded, structure=kernel)
    return opened



def morph_closing(binary_image, kernel_size=3):
    """
    Thực hiện phép Closing (Dilation -> Erosion)[cite: 124].

    Args:
        binary_image (np.array): Ảnh nhị phân (True/False hoặc 1/0).
        kernel_size (int): Kích thước của kernel.

    Returns:
        np.array: Ảnh nhị phân sau khi Closing.
    """
    kernel = np.ones((kernel_size, kernel_size))

    # # 1. Dilation (Giãn)
    # dilated = custom_dilation(image_int, kernel)
    # # 2. Erosion (Co)
    # closed = custom_erosion(dilated, kernel)
    #
    # return closed.astype(bool)

    # 1. Dilation (Giãn)
    dilated = ndimage.binary_dilation(binary_image, structure=kernel)
    # 2. Erosion (Co)
    closed = ndimage.binary_erosion(dilated, structure=kernel)
    return closed


if __name__ == "__main__":
    print("--- KIỂM TRA src/ip_utils.py ---")

    # 1. Thử tạo 1 ảnh xám giả
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[25:75, 25:75] = 150  # Một hình vuông màu xám ở giữa
    test_image[10:20, 10:20] = 255  # Một ít nhiễu trắng
    test_image = test_image + 20  # Thêm nhiễu nền

    print(f"Ảnh test có kích thước: {test_image.shape}")

    # 2. Thử Otsu
    threshold = otsu_threshold(test_image)
    print(f"Ngưỡng Otsu tự tính: {threshold}")

    # Tạo ảnh nhị phân
    binary = test_image > threshold

    # 3. Thử Opening
    opened = morph_opening(binary, kernel_size=5)

    # 4. Thử Closing
    closed = morph_closing(binary, kernel_size=5)

    print(f"Chạy Otsu (kết quả nhị phân): {binary.shape}")
    print(f"Chạy Opening (kết quả): {opened.shape}")
    print(f"Chạy Closing (kết quả): {closed.shape}")

    print("--- HOÀN THÀNH KIỂM TRA IP UTILS ---")

