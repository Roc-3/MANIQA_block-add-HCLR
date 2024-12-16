# 添加resnet
import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
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
        # before normalization and transpose, convert to gray image
        d_img_gray = cv2.cvtColor(d_img_org, cv2.COLOR_RGB2GRAY)# (224, 224, 3)
        d_img_org = np.array(d_img_org).astype('float32') / 255 
        d_img_org = np.transpose(d_img_org, (2, 0, 1))
    
        # texture
        _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
        d_img_texture = d_img_texture.astype('float32') / 255 
        d_img_texture = np.expand_dims(d_img_texture, axis=0) 
        d_img_texture = np.repeat(d_img_texture, 3, axis=0)  # (3, 224, 224)
    
        if self.transform:  # random crop, flip
            d_img_org = self.transform(d_img_org)
            d_img_texture = self.transform(d_img_texture)
        if self.normalize: # vit normalization
            d_img_org = self.normalize(d_img_org)
        # 将 d_img_texture 转换为 torch.Tensor
        d_img_texture = torch.tensor(d_img_texture, dtype=torch.float32)
        # ResNet normalization for d_img_texture
        normalize_resnet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        d_img_texture = normalize_resnet(d_img_texture)
    
        score = self.data_dict['score_list'][idx]
        score = torch.from_numpy(np.array(score)).type(torch.FloatTensor)
    
        sample = {
            'd_img_org': d_img_org,
            'd_img_texture': d_img_texture,
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
    
    def visualize_and_save(self, d_img_org, d_img_texture, idx):
        # 将图像从 Tensor 转换为 NumPy 数组以进行可视化
        d_img_org = d_img_org.numpy().transpose((1, 2, 0))
        d_img_texture = d_img_texture.numpy()[0]  # 只取第一个颜色通道

        # 可视化原图和纹理图像
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Transformed Original Image')
        plt.imshow(d_img_org)

        plt.subplot(1, 2, 2)
        plt.title('Transformed Texture Image')
        plt.imshow(d_img_texture, cmap='gray')

        # 保存图像
        save_path = f"./transformed_images/sample_{idx}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    # def __getitem__(self, idx):
    #     d_img_name = self.data_dict['d_img_list'][idx]
    #     d_img_name = d_img_name.encode('utf-8').decode('utf-8-sig')
    #     d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
    
    #     if(d_img.shape[0] > d_img.shape[1]):
    #         print('there is a image with height > width named: ', d_img_name)
    #     d_img_org = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #     d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
    #     # before normalization and transpose, convert to gray image
    #     d_img_gray = cv2.cvtColor(d_img_org, cv2.COLOR_RGB2GRAY)# (224, 224, 3)
    #     # cv2.imwrite('d_img_org.png', d_img_org)
    #     d_img_org = np.array(d_img_org).astype('float32') / 255 
    #     d_img_org = np.transpose(d_img_org, (2, 0, 1))
    
    #     # texture
    #     _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
    #     # cv2.imwrite('texture_image.png', d_img_texture)
    #     d_img_texture = d_img_texture.astype('float32') / 255 
    #     d_img_texture = np.expand_dims(d_img_texture, axis=0) 
    #     d_img_texture = np.repeat(d_img_texture, 3, axis=0) # (3, 224, 224)

    #     if self.transform: # random crop, flip
    #         d_img_org = self.transform(d_img_org)
    #         d_img_texture = self.transform(d_img_texture)
    #     if self.normalize: # vit normalization
    #         d_img_org = self.normalize(d_img_org)
    #         # d_img_texture = self.normalize(d_img_texture)
    #         normalize_resnet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         d_img_texture = normalize_resnet(d_img_texture)

    #     score = self.data_dict['score_list'][idx]
    #     score = torch.from_numpy(np.array(score)).type(torch.FloatTensor)

    #     sample = {
    #         'd_img_org': d_img_org,
    #         'd_img_texture': torch.tensor(d_img_texture, dtype=torch.float32),
    #         'score': score
    #     }
    #     # print('sample d_img_org:', sample['d_img_org'].shape) # (3, 224, 224)
    #     # print('sample d_img_texture:', sample['d_img_texture'].shape) # (3, 224, 224)

    #     return sample

# import os
# import torch
# import numpy as np
# import cv2
# import torch.nn.functional as F

# from PIL import Image

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

#         self.data_dict = {'d_img_list': dis_files_data, 'd_img_texture': d_img_texture, 'score_list': score_data}

#     def normalization(self, data):
#         range = np.max(data) - np.min(data)
#         return (data - np.min(data)) / range

#     def __len__(self):
#         return len(self.data_dict['d_img_list'])
    
#     def __getitem__(self, idx):
#         d_img_name = self.data_dict['d_img_list'][idx]
#         d_img_name = d_img_name.encode('utf-8').decode('utf-8-sig')
#         d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
    
#         if(d_img.shape[0] > d_img.shape[1]):
#             print('there is a image with height > width named: ', d_img_name)
#         d_img_org = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
#         d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
#         # before normalization and transpose, convert to gray image
#         d_img_gray = cv2.cvtColor(d_img_org, cv2.COLOR_RGB2GRAY)
#         d_img_gray = (d_img_gray - np.min(d_img_gray)) / (np.max(d_img_gray) - np.min(d_img_gray) + 1e-8) # (224, 224)
#         d_img_org = np.array(d_img_org).astype('float32') / 255 # (224, 224, 3)
#         d_img_org = np.transpose(d_img_org, (2, 0, 1))
    
#         # texture
#         _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
#         d_img_texture = d_img_texture.astype('float32')
#         d_img_texture = (d_img_texture - np.min(d_img_texture)) / (np.max(d_img_texture) - np.min(d_img_texture) + 1e-8)  
#         d_img_texture = np.expand_dims(d_img_texture, axis=0) 
#         d_img_texture = np.repeat(d_img_texture, 3, axis=0) # (3, 224, 224)

#         if self.transform: # random crop, flip
#             d_img_org = self.transform(d_img_org)
#             d_img_texture = self.transform(d_img_texture)
#         if self.normalize: # vit normalization
#             d_img_org = self.normalize(d_img_org)

#         score = self.data_dict['score_list'][idx]
#         score = torch.from_numpy(np.array(score)).type(torch.FloatTensor)

#         sample = {
#             'd_img_org': d_img_org,
#             'd_img_texture': d_img_texture,
#             'score': score
#         }
#         # print('sample d_img_org:', sample['d_img_org'].shape) # (3, 224, 224)
#         # print('sample d_img_texture:', sample['d_img_texture'].shape) # (3, 224, 224)

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