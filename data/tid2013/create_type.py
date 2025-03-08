import os

# 定义失真类型
distortion_types = [
    "Additive Gaussian noise",
    "Additive noise in color components is more intensive than additive noise in the luminance component",
    "Spatially correlated noise",
    "Masked noise",
    "High frequency noise",
    "Impulse noise",
    "Quantization noise",
    "Gaussian blur",
    "Image denoising",
    "JPEG compression",
    "JPEG2000 compression",
    "JPEG transmission errors",
    "JPEG2000 transmission errors",
    "Non eccentricity pattern noise",
    "Local block-wise distortions of different intensity",
    "Mean shift (intensity shift)",
    "Contrast change",
    "Change of color saturation",
    "Multiplicative Gaussian noise",
    "Comfort noise",
    "Lossy compression of noisy images",
    "Image color quantization with dither",
    "Chromatic aberrations",
    "Sparse sampling and reconstruction"
]

# 创建保存失真类型文件的文件夹
output_dir = 'data/tid2013/type'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化字典来存储每种失真类型的图像信息
distortion_files = {i: [] for i in range(1, 25)}

# 读取原始标签文件并按失真类型划分
with open('data/tid2013/tid2013_label.txt', 'r') as file:
    for line in file:
        score, image_name = line.strip().split()
        distortion_type = int(image_name.split('_')[1])
        distortion_files[distortion_type].append(f"{score} {image_name}")

# 将每种失真类型的图像信息写入对应的文件
for i in range(1, 25):
    output_file = os.path.join(output_dir, f"type_{i}.txt")
    with open(output_file, 'w') as file:
        file.write('\n'.join(distortion_files[i]))

print("文件划分完成并保存到 'data/tid2013/type' 文件夹中。")