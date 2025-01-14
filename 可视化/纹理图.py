import os
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def gaussian_filter(self, image, sigma):
        """
        对图像应用高斯滤波器
        """
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    def structure_texture_decomposition(self, image, sigma):
        """
        使用高斯滤波器进行结构-纹理分解
        :param image: 输入图像
        :param sigma: 高斯滤波器的标准差
        :return: 结构图像和纹理图像
        """
        # 计算结构图像（高斯模糊）
        structure_image = self.gaussian_filter(image, sigma)
        
        # 计算纹理图像（原图减去结构图像）
        texture_image = image - structure_image
        
        return structure_image, texture_image

    def process_and_save_texture(self, image_path, sigma, output_folder):
        """
        处理输入图像并保存纹理图像
        :param image_path: 输入图像路径
        :param sigma: 高斯滤波器的标准差
        :param output_folder: 输出文件夹路径
        """
        # 读取输入图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return

        # 第一种：对原图进行纹理图变化之后再resize 224x224
        _, texture_image_1 = self.structure_texture_decomposition(img, sigma)
        texture_image_1_resized = cv2.resize(texture_image_1, (224, 224), interpolation=cv2.INTER_CUBIC)

        # 第二种：先resize成224x224再变成纹理图
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        _, texture_image_2 = self.structure_texture_decomposition(img_resized, sigma)

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 保存第一种纹理图像
        output_path_1 = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_texture1.png')
        cv2.imwrite(output_path_1, texture_image_1_resized)
        print(f"纹理图像已保存: {output_path_1}")

        # 保存第二种纹理图像
        output_path_2 = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_texture2.png')
        cv2.imwrite(output_path_2, texture_image_2)
        print(f"纹理图像已保存: {output_path_2}")

        # 保存调整大小后的原始图像
        output_path_resized = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_resized.png')
        cv2.imwrite(output_path_resized, img_resized)
        print(f"调整大小后的原始图像已保存: {output_path_resized}")

if __name__ == "__main__":
    # 示例用法
    input_image_path = "可视化/koniq.png"
    output_directory = "可视化/纹理图"
    sigma_value = 2.0  # 高斯滤波器的标准差

    processor = ImageProcessor()
    processor.process_and_save_texture(input_image_path, sigma_value, output_directory)