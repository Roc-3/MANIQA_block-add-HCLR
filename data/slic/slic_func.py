# # -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import os

from data.slic.utils import local_normalize

import numpy as np
import cv2
import os
from PIL import Image
import numpy as np

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

    def visualize_segments(self, slic, selected_superpixels, save_path='slic_vis.png'):
        vis_img = self.img.copy()
        mask = slic.getLabelContourMask()
        vis_img[mask == 255] = [255, 0, 0]  # 将所有超像素边缘标记为红色

        for sp_id in selected_superpixels:
            loc = np.where(self.label == sp_id)
            selected_indices = np.linspace(0, len(loc[0]) - 1, self.patch_n_nodes, dtype=int)
            vis_img[loc[0][selected_indices], loc[1][selected_indices]] = [0, 255, 0]  # 将选取的像素点标记为绿色

        # 保存可视化结果
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        vis_img = vis_img / 255.0
        plt.imsave(save_path, vis_img)

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
            print('img name and its number of superpixels:', save_path, self.num_clusters)

            # Get all superpixels and remove those with fewer nodes
            all_clusters = list(range(self.num_clusters))
            cleaned_clusters = self.remove_cluster(clusters=all_clusters)

            # Select the superpixels using np.linspace
            selected_superpixels = np.linspace(0, len(cleaned_clusters) - 1, self.image_n_nodes, dtype=int)
            selected_superpixels = [cleaned_clusters[i] for i in selected_superpixels]

            # Extract pixel color information from the selected superpixels
            superpixel_data = {}
            for sp_id in selected_superpixels:
                # Get the locations of the pixels in this superpixel
                loc = np.where(self.label == sp_id)

                # Get pixel colors (R, G, B)
                pixel_colors = self.img[loc]

                # Select patch_n_nodes pixels using np.linspace
                num_pixels = len(pixel_colors)

                # Use np.linspace to select indices evenly
                selected_indices = np.linspace(0, num_pixels - 1, self.patch_n_nodes, dtype=int)
                selected_pixels = pixel_colors[selected_indices]

                # Sort selected pixels by their original positions
                positions = np.stack((loc[0][selected_indices], loc[1][selected_indices]), axis=-1)
                sorted_indices = np.argsort(positions[:, 0] * self.img.shape[1] + positions[:, 1])
                sorted_pixels = selected_pixels[sorted_indices]

                superpixel_data[sp_id] = sorted_pixels

            # Save the selected superpixel data
            np.save(save_path, {'superpixel_data': superpixel_data})

        # Convert the superpixel data into a tensor with shape (image_n_nodes, patch_n_nodes, channel)
        superpixel_tensor = np.stack([superpixel_data[sp_id] for sp_id in superpixel_data], axis=0)

        # # visualize
        # if slic is not None:
        #     self.visualize_segments(slic, selected_superpixels, save_path='slic_vis.png')
        # ############################

        return superpixel_tensor
        
    def remove_cluster(self, clusters):
        """
        Remove the clusters with fewer nodes (than patch_n_nodes).
        :param clusters: List of superpixel IDs
        :return: List of superpixel IDs with sufficient pixels
        """
        cleaned_clusters = []
        for sp_id in clusters:
            # Get the pixel locations for the superpixel
            loc = np.where(self.label == sp_id)

            # Check if the number of pixels in this superpixel is sufficient
            if len(loc[0]) >= self.patch_n_nodes:
                cleaned_clusters.append(sp_id)

        return cleaned_clusters
