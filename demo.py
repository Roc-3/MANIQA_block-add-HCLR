from pyimgsaliency.saliency import get_saliency_rbd
import cv2
import os
import tempfile
from utils.rtv2saliency.rtv2 import tsmooth
from tqdm import tqdm
import numpy as np

# 输入文件夹路径
input_folder = '../all_dataset/LIVEC/Images/'
output_folder = 'slic_sal'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 支持的图像扩展名
supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

# 获取输入文件夹中的所有文件
file_list = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in supported_extensions)]

# 遍历输入文件夹中的所有文件
for filename in tqdm(file_list, desc="Processing images"):
    input_path = os.path.join(input_folder, filename)
    
    # 确定保存格式为 JPG
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
    
    # 检查文件是否已经存在
    if os.path.exists(output_path):
        continue
    
    # 读取图像
    d_img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # 224x224 大小的图像
    d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # 去纹理操作
    rtv = tsmooth(d_img, maxIter=2)  # 使用去纹理方法
    
    # 将去纹理图像保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, rtv)
        temp_file_path = temp_file.name
    
    # 计算 RBD 显著性图
    try:
        rbd = get_saliency_rbd(temp_file_path).astype('uint8')
    except IndexError as e:
        print(f"Error processing {filename}: {e}")
        print(f"Image shape: {rtv.shape}")
        print(f"Image dtype: {rtv.dtype}")
        continue

    # 删除临时文件
    os.remove(temp_file_path)
    
    # 调整显著性图大小为 224x224
    rbd_resized = cv2.resize(rbd, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # 保存 RBD 显著性图为 JPG 格式
    cv2.imwrite(output_path, rbd_resized)
    print(f'Saved RBD saliency map for {filename} to {output_path}')

# 计算并输出文件夹中的文件个数
file_count = len([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])
print(f'Total number of files in {output_folder}: {file_count}')