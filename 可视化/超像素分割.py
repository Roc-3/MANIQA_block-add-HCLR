import cv2
import numpy as np
from PIL import Image
import os
from utils import local_normalize
import matplotlib.pyplot as plt

class SLIC:
    """
    SLIC Superpixel Segmentation Algorithm
    """
    def __init__(self, img, args):
        self.img = np.array(img)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.normalize_img = local_normalize(img=self.img, num_ch=3, const=127.0)
        self.image_n_nodes = args['image_n_nodes']
        self.patch_n_nodes = args['patch_n_nodes']
        self.region_size = args['region_size']
        self.ruler = args['ruler']
        self.iterate = args['iterate']

    def slic_function(self, save_path='', visualize_path='slic_save'):
        if save_path and os.path.exists(save_path):
            slic_res = np.load(save_path, allow_pickle=True).item()
            superpixel_data = slic_res['superpixel_data']
        else:
            # Perform SLIC segmentation
            slic = cv2.ximgproc.createSuperpixelSLIC(image=self.img_lab, region_size=self.region_size, ruler=self.ruler)
            slic.iterate(self.iterate)
    
            # Get labels and number of superpixels
            self.label = slic.getLabels()
            self.num_clusters = slic.getNumberOfSuperpixels()

            """输出最大和最小的超像素内部像素点的个数和所有的超像素个数"""
            superpixel_sizes = [len(np.where(self.label == sp_id)[0]) for sp_id in range(self.num_clusters)]
            max_superpixel_size = max(superpixel_sizes)
            min_superpixel_size = min(superpixel_sizes)
            mean_superpixel_size = np.mean(superpixel_sizes)
            total_superpixels = len(superpixel_sizes)
            print(f"Max superpixel size: {max_superpixel_size}")
            print(f"Min superpixel size: {min_superpixel_size}")
            print(f"Mean superpixel size: {mean_superpixel_size}")
            print(f"Total number of superpixels: {total_superpixels}")
            """"""

            # Select the superpixels using np.linspace
            all_clusters = list(range(self.num_clusters))
            non_empty_clusters = [sp_id for sp_id in all_clusters if np.any(self.label == sp_id)]
            
            if len(non_empty_clusters) < self.image_n_nodes:
                selected_superpixels = np.random.choice(non_empty_clusters, self.image_n_nodes, replace=True)
            else:
                selected_superpixels = np.linspace(0, len(non_empty_clusters) - 1, self.image_n_nodes, dtype=int)
                selected_superpixels = [non_empty_clusters[i] for i in selected_superpixels]
    
            # Extract pixel color information from the selected superpixels
            superpixel_data = {}
            for new_sp_id, sp_id in enumerate(selected_superpixels):
                # Get the locations of the pixels in this superpixel
                loc = np.where(self.label == sp_id)
    
                # Get pixel colors (R, G, B)
                pixel_colors = self.img[loc]
    
                # Select patch_n_nodes pixels using np.linspace
                num_pixels = len(pixel_colors)
    
                if num_pixels == 0:
                    continue
    
                if num_pixels < self.patch_n_nodes:
                    selected_indices = np.random.choice(num_pixels, self.patch_n_nodes, replace=True)
                else:
                    selected_indices = np.linspace(0, num_pixels - 1, self.patch_n_nodes, dtype=int)
    
                selected_pixels = pixel_colors[selected_indices]
    
                # Sort selected pixels by their original positions
                positions = np.stack((loc[0][selected_indices], loc[1][selected_indices]), axis=-1)
                sorted_indices = np.argsort(positions[:, 0] * self.img.shape[1] + positions[:, 1])
                sorted_pixels = selected_pixels[sorted_indices]
    
                superpixel_data[new_sp_id] = sorted_pixels
    
            while len(superpixel_data) < self.image_n_nodes:
                missing_sp_id = len(superpixel_data)
                random_sp_id = np.random.choice(non_empty_clusters)
                loc = np.where(self.label == random_sp_id)
                pixel_colors = self.img[loc]
                num_pixels = len(pixel_colors)
                if num_pixels == 0:
                    continue
                if num_pixels < self.patch_n_nodes:
                    selected_indices = np.random.choice(num_pixels, self.patch_n_nodes, replace=True)
                else:
                    selected_indices = np.linspace(0, num_pixels - 1, self.patch_n_nodes, dtype=int)
                selected_pixels = pixel_colors[selected_indices]
                positions = np.stack((loc[0][selected_indices], loc[1][selected_indices]), axis=-1)
                sorted_indices = np.argsort(positions[:, 0] * self.img.shape[1] + positions[:, 1])
                sorted_pixels = selected_pixels[sorted_indices]
                superpixel_data[missing_sp_id] = sorted_pixels
    
            # Save the selected superpixel data
            # np.save(save_path, {'superpixel_data': superpixel_data})
    
            # Visualize the selected superpixels with red edges and green selected points
            vis_img_red_green = self.img.copy()
            mask = slic.getLabelContourMask()
            vis_img_red_green[mask == 255] = [255, 0, 0]  # 将所有超像素边缘标记为红色

            for sp_id in selected_superpixels:
                loc = np.where(self.label == sp_id)
                selected_indices = np.linspace(0, len(loc[0]) - 1, self.patch_n_nodes, dtype=int)
                vis_img_red_green[loc[0][selected_indices], loc[1][selected_indices]] = [0, 255, 0]  # 将选取的像素点标记为绿色

            vis_img_red_green = cv2.cvtColor(vis_img_red_green, cv2.COLOR_BGR2RGB)
            vis_img_red_green = vis_img_red_green / 255.0
            plt.imsave(os.path.join(visualize_path, 'slic_vis_red_green.png'), vis_img_red_green)
            print('The image with red edges and green selected points has been saved:', os.path.join(visualize_path, 'slic_vis_red_green.png'))

            # Visualize the selected superpixels with dilated white edges
            vis_img_white_edge = self.img.copy()
            mask = slic.getLabelContourMask()
            kernel = np.ones((3, 3), np.uint8)  # 定义一个3x3的核
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # 扩展边缘
            vis_img_white_edge[dilated_mask == 255] = [255, 255, 255]  # 将扩展后的边缘标记为白色

            vis_img_white_edge = cv2.cvtColor(vis_img_white_edge, cv2.COLOR_BGR2RGB)
            vis_img_white_edge = vis_img_white_edge / 255.0
            plt.imsave(os.path.join(visualize_path, 'slic_vis_white_edge.png'), vis_img_white_edge)
            print('The image with dilated white edges has been saved:', os.path.join(visualize_path, 'slic_vis_white_edge.png'))

        # Convert the superpixel data into a tensor with shape (image_n_nodes, patch_n_nodes, channel)
        superpixel_tensor = np.stack([superpixel_data[sp_id] for sp_id in range(self.image_n_nodes) if sp_id in superpixel_data], axis=0)
    
        if superpixel_tensor.shape != (self.image_n_nodes, self.patch_n_nodes, 3):
            print('image still not compete:', save_path)
            superpixel_tensor = np.resize(superpixel_tensor, (self.image_n_nodes, self.patch_n_nodes, 3))
    
        return superpixel_tensor
    
class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, d_img):
        # r_img : C x H x W (numpy)

        # d_img = sample['d_img_org']
        # score = sample['score']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        
        # For koniq10k
        if h == new_h and w == new_w:
            ret_d_img = d_img
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            ret_d_img = d_img[:, top: top + new_h, left: left + new_w]

        # sample = {
        #     'd_img_org': ret_d_img,
        #     'score': score
        # }
        # return sample
         
        return ret_d_img
    
def main():
    # 图像路径
    img_path = '可视化/spaq.png'
    save_path = '220.npy'
    visualize_path = '可视化'

    # 加载图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"图像 {img_path} 加载失败！")
        return

    # 下采样到 500x500
    img_down = cv2.resize(img, (512, 384), interpolation=cv2.INTER_CUBIC)

    # 随机裁剪图像到 224x224
    crop_size = 224
    rand_crop = RandCrop(crop_size)
    img_cropped = rand_crop(np.transpose(img, (2, 0, 1)))  # 转换为 C x H x W
    img_cropped = np.transpose(img_cropped, (1, 2, 0))  # 转换回 H x W x C

    # SLIC参数
    slic_args = {   
        'image_n_nodes': 125, 
        'patch_n_nodes': 600,
        'region_size': 40,
        'ruler': 10.0,
        'iterate': 10
    }
    # livec setting
    # slic_args = {
    #     'image_n_nodes': 140,
    #     'patch_n_nodes': 600,
    #     'region_size': 40,
    #     'ruler': 10.0,
    #     'iterate': 10
    # }
    
    # 执行SLIC分割
    slic = SLIC(img=img_down, args=slic_args)
    superpixel_tensor = slic.slic_function(save_path=save_path, visualize_path=visualize_path)

    print(f"分割完成！超像素张量形状: {superpixel_tensor.shape}")

if __name__ == '__main__':
    main()