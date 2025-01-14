import os
import cv2
import numpy as np

def split_image_into_nine(image_path, output_folder):
    # 读取输入图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 获取图像尺寸
    h, w, _ = img.shape
    h_step = h // 3
    w_step = w // 3

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 分割图像并保存
    for i in range(3):
        for j in range(3):
            top = i * h_step
            left = j * w_step
            bottom = (i + 1) * h_step
            right = (j + 1) * w_step
            img_crop = img[top:bottom, left:right]
            output_path = os.path.join(output_folder, f"crop_{i}_{j}.png")
            cv2.imwrite(output_path, img_crop)
            print(f"保存分割图像: {output_path}")

if __name__ == "__main__":
    # 示例用法
    input_image_path = "可视化/787.JPG"
    output_directory = "可视化/分割9块"
    split_image_into_nine(input_image_path, output_directory)