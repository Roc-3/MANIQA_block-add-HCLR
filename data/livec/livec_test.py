import os
import torch
import numpy as np
import cv2 

from utils.process import RandCrop, ToTensor, Normalize, five_point_crop

from utils.slic.slic_func import SLIC

class LIVEC(torch.utils.data.Dataset):
    def __init__(self, dis_path):
        super(LIVEC, self).__init__()
        self.dis_path = dis_path
        self.normalize =Normalize(0.5, 0.5)

        dis_files_data = []
        for dis in os.listdir(dis_path):
            if dis.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                dis_files_data.append(dis)
        self.data_dict = {'d_img_list': dis_files_data}

        # slic setting
        self.slic_args = {
            'image_n_nodes': 140,
            'patch_n_nodes': 600,
            'region_size': 40,
            'ruler': 10.0,
            'iterate': 10
        }

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_name = d_img_name.encode('utf-8').decode('utf-8-sig')
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        if d_img is None:
            raise ValueError("图像加载失败: ", d_img_name)

        
        # slic
        d_img_slic = cv2.resize(d_img, (500, 500), interpolation=cv2.INTER_CUBIC)
        d_img_slic = np.array(d_img_slic).astype('uint8') # hwc
        # slic superpixel
        ############################################
        save_dir = 'slic_livec'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{os.path.splitext(d_img_name)[0]}_seg.npy')

        slic_class = SLIC(img=d_img_slic, args=self.slic_args)
        d_img_slic = slic_class.slic_function(save_path=save_path, visualize_path='visual_path') 
        d_img_slic = d_img_slic.astype('float32') / 255 # (image_n_nodes, patch_n_nodes, 3)
        ############################################

        # vit img 
        d_img_vit = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img_vit = cv2.cvtColor(d_img_vit, cv2.COLOR_BGR2RGB)
        d_img_vit = np.array(d_img_vit).astype('float32') / 255
        d_img_vit = np.transpose(d_img_vit, (2, 0, 1))
        if self.normalize: # vit normalization
            d_img_vit = self.normalize(d_img_vit) # (3, 224, 224)

        # texture img 
        d_img_gray = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
        _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
        d_img_texture = d_img_texture.astype('float32') / 255
        d_img_texture = np.expand_dims(d_img_texture, axis=0)
        d_img_texture = np.repeat(d_img_texture, 3, axis=0)# (3, 500, 500)
        d_img_texture = np.transpose(d_img_texture, (1, 2, 0))# (500, 500, 3)
        d_img_texture = cv2.resize(d_img_texture, (224, 224), interpolation=cv2.INTER_CUBIC)# (224, 224, 3)
        d_img_texture = torch.tensor(d_img_texture, dtype=torch.float32)
        d_img_texture = np.transpose(d_img_texture, (2, 0, 1))

        sample = {
            'd_img_org': d_img_vit,
            'd_img_texture': d_img_texture,
            'd_img_slic': d_img_slic,
            'd_name': d_img_name
        }

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