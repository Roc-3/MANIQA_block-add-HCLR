import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from all_code.TranSalNet.TranSalNet_Res import TranSalNet
from data.slic.slic_func import SLIC

class LIVEC(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, normalize, keep_ratio):
        super(LIVEC, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.normalize = normalize

        dis_files_data, d_img_texture, score_data = [], [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

        # slic setting
        self.slic_args = {
            'image_n_nodes': 200,
            'patch_n_nodes': 300,
            'region_size': 28,
            'ruler': 10.0,
            'iterate': 10
        }

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_name = d_img_name.encode('utf-8').decode('utf-8-sig')
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
    
        if(d_img.shape[0] > d_img.shape[1]):
            print('there is a image with height > width named: ', d_img_name)
        d_img_org = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
        
        # before normalization to gray
        d_img_gray = cv2.cvtColor(d_img_org, cv2.COLOR_RGB2GRAY)# (224, 224, 3)
        # use 500 500 size to get slic
        d_img_slic = cv2.resize(d_img, (500, 500), interpolation=cv2.INTER_CUBIC)
        d_img_slic = cv2.cvtColor(d_img_slic, cv2.COLOR_BGR2RGB)

        d_img_org = np.array(d_img_org).astype('float32') / 255 
        d_img_org = np.transpose(d_img_org, (2, 0, 1))
        d_img_slic = np.array(d_img_slic).astype('float32')  #不进行归一化
        d_img_slic = np.transpose(d_img_slic, (2, 0, 1))

        # texture
        _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
        d_img_texture = d_img_texture.astype('float32') / 255 
        d_img_texture = np.expand_dims(d_img_texture, axis=0) 
        d_img_texture = np.repeat(d_img_texture, 3, axis=0)  # (3, 224, 224)

        if self.transform:  # random crop, flip
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            d_img_org = self.transform(d_img_org)
            torch.manual_seed(seed)
            d_img_texture = self.transform(d_img_texture)
            torch.manual_seed(seed)
            d_img_slic = self.transform(d_img_slic)
            # visualize_and_save(d_img_org, d_img_texture, d_img_name)

        # slic superpixel
        ############################################
        save_dir = 'slic_save'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{os.path.splitext(d_img_name)[0]}_seg.npy')

        d_img_slic = d_img_slic.transpose(1, 2, 0)  # Convert to (H, W, C) for SLIC

        slic_class = SLIC(img=d_img_slic, args=self.slic_args)
        d_img_slic = slic_class.slic_function(save_path=save_path, visualize_path='visual_path') # (image_n_nodes, image_n_nodes, 3)

        ############################################

        if self.normalize: # vit normalization
            d_img_org = self.normalize(d_img_org)
        d_img_texture = torch.tensor(d_img_texture, dtype=torch.float32)
    
        score = self.data_dict['score_list'][idx]
        score = torch.from_numpy(np.array(score)).type(torch.FloatTensor)
    
        sample = {
            'd_img_org': d_img_org,
            'd_img_texture': d_img_texture,
            'd_img_slic': d_img_slic,
            'score': score
        }
        # print('sample d_img_org:', sample['d_img_org'].shape)  # (3, 224, 224)
        # print('sample d_img_texture:', sample['d_img_texture'].shape)  # (3, 224, 224)
    
        return sample
    
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
    
def visualize_and_save(d_img_org, d_img_texture, img_name):
    """
    Visualize and save the original image and texture image side by side.

    Parameters:
    - d_img_org: The original image (normalized and transposed to (3, H, W))
    - d_img_texture: The texture image (normalized and transposed to (3, H, W))
    - img_name: Name of the image (used for saving the visualized output)
    """
    # Convert the tensors back to image format for visualization
    d_img_org_vis = (d_img_org.transpose(1, 2, 0) * 255).astype(np.uint8)  # Convert to (H, W, 3)
    d_img_texture_vis = (d_img_texture.transpose(1, 2, 0) * 255).astype(np.uint8)  # Convert to (H, W, 3)

    # Concatenate images horizontally
    combined_img = np.concatenate((d_img_org_vis, d_img_texture_vis), axis=1)

    # Save the combined image
    save_path = os.path.join('./', f"visualized_{img_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    print(f"Visualized image saved at: {save_path}")

# class LIVEC(torch.utils.data.Dataset):
#     def __init__(self, dis_path, txt_file_name, list_name, transform, normalize, keep_ratio):
#         super(LIVEC, self).__init__()
#         self.dis_path = dis_path
#         self.txt_file_name = txt_file_name
#         self.transform = transform
#         self.normalize = normalize

#         dis_files_data, d_img_texture, score_data = [], [], []
#         with open(self.txt_file_name, 'r') as listFile:
#             for line in listFile:
#                 dis, score = line.split()
#                 if dis in list_name:
#                     score = float(score)
#                     dis_files_data.append(dis)
#                     score_data.append(score)

#         # reshape score_list (1xn -> nx1)
#         score_data = np.array(score_data)
#         score_data = self.normalization(score_data)
#         score_data = list(score_data.astype('float').reshape(-1, 1))

#         self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

#     def normalization(self, data):
#         range = np.max(data) - np.min(data)
#         return (data - np.min(data)) / range

#     def __len__(self):
#         return len(self.data_dict['d_img_list'])
    
#     def __getitem__(self, idx):
#         d_img_name = self.data_dict['d_img_list'][idx]
#         d_img_name = d_img_name.encode('utf-8').decode('utf-8-sig')
#         d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
    
#         d_img_org = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
#         d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
#         d_img_org = np.array(d_img_org).astype('float32') / 255 
#         d_img_org = np.transpose(d_img_org, (2, 0, 1)) # chw
        
#         d_img_gray = cv2.resize(d_img, (500, 500), interpolation=cv2.INTER_CUBIC)
#         d_img_gray = cv2.cvtColor(d_img_gray, cv2.COLOR_BGR2GRAY)
#         _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
#         d_img_texture = d_img_texture.astype('float32') / 255 
#         d_img_texture = np.expand_dims(d_img_texture, axis=0)  
#         d_img_texture = np.repeat(d_img_texture, 3, axis=0) 

#         d_img_sal = self.preprocess_img(os.path.join(self.dis_path, d_img_name), 288, 384, 3)  # padding and resizing input image into 384x288
#         d_img_sal = np.array(d_img_sal) / 255.  # 将 numpy 数组归一化
#         d_img_sal = np.transpose(d_img_sal, (2, 0, 1))  # chw

#         if self.transform:  # random crop, flip
#             d_img_org = self.transform(d_img_org)
#             d_img_texture = self.transform(d_img_texture)
#             d_img_sal = self.transform(d_img_sal)
        
#         if self.normalize: # vit normalization
#             d_img_org = self.normalize(d_img_org)
#             # d_img_texture = self.normalize(d_img_texture) # 纹理图不需要进行归一化
#             d_img_sal = self.normalize(d_img_sal)
    
#         score = self.data_dict['score_list'][idx]
#         score = torch.from_numpy(np.array(score)).type(torch.FloatTensor)
    
#         sample = {
#             'd_img_org': d_img_org,
#             # 'd_imgs_texture': d_img_texture,
#             'd_img_sal': d_img_sal,
#             'score': score
#         }
#         # print('sample d_img_org:', sample['d_img_org'].shape)  # (3, 224, 224)
#         # print('sample d_img_texture:', sample['d_img_texture'].shape)  # (3, 224, 224)
    
#         return sample
    
#     def gaussian_filter(self, image, sigma):
#         """
#         对图像应用高斯滤波器
#         """
#         return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

#     def structure_texture_decomposition(self, image, sigma):
#         """
#         使用高斯滤波器进行结构-纹理分解
#         :param image: 输入图像
#         :param sigma: 高斯滤波器的标准差
#         :return: 结构图像和纹理图像
#         """
#         # 计算结构图像（高斯模糊）
#         structure_image = self.gaussian_filter(image, sigma)
        
#         # 计算纹理图像（原图减去结构图像）
#         texture_image = image - structure_image
        
#         return structure_image, texture_image
    
#     # saliency map preprocess
#     def preprocess_img(self, img_dir, shape_r=288, shape_c=384, channels=3): #保持纵横比，指定288x384大小，居中填充
#         if channels == 1:
#             img = cv2.imread(img_dir, 0)
#         elif channels == 3:
#             img = cv2.imread(img_dir)

#         img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
#         if channels == 1:
#             img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
#         original_shape = img.shape
#         rows_rate = original_shape[0] / shape_r
#         cols_rate = original_shape[1] / shape_c
#         if rows_rate > cols_rate:
#             new_cols = (original_shape[1] * shape_r) // original_shape[0]
#             img = cv2.resize(img, (new_cols, shape_r))
#             if new_cols > shape_c:
#                 new_cols = shape_c
#             img_padded[:,
#             ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
#         else:
#             new_rows = (original_shape[0] * shape_c) // original_shape[1]
#             img = cv2.resize(img, (shape_c, new_rows))

#             if new_rows > shape_r:
#                 new_rows = shape_r
#             img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
#             :] = img

#         return img_padded