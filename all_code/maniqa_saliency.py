"""
在输入vit前使用saliency和rgb进行输入图像重组融合
"""
import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
# from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
from models.resnet import ResBlockGroup
from models.hclr import HCLR
from models.hclr import Attention
from models.newDyD import DynamicDWConv
import torch.nn.functional as F

import torch
import numpy as np
import cv2
from all_code.TranSalNet.TranSalNet_Res import TranSalNet
class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x

class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        # tab1
        ######################################################################
        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        # vit block1
        ######################################################################
        self.block1 = Block(
            dim=embed_dim,
            num_heads=num_heads[0],
            mlp_ratio=dim_mlp / embed_dim,
            drop=drop,
        )
        # resblock1
        ######################################################################
        self.resblock1 = ResBlockGroup(embed_dim, num_blocks=2)

        # tab2
        ######################################################################
        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        # vit block2
        ######################################################################
        self.block2 = Block(
            dim=embed_dim // 2,
            num_heads=num_heads[1],
            mlp_ratio=dim_mlp / (embed_dim // 2),
            drop=drop,
        )     
        # resblock2
        ######################################################################
        self.resblock2 = ResBlockGroup(embed_dim // 2, num_blocks=2)
        
        # reduce dim
        ######################################################################
        self.catconv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0) # stage1
        self.catconv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # stage2

        # output_dckg
        self.dyd_conv = nn.Conv2d(3, embed_dim // 2, 3, 1, 1)
        self.dyd_0= DynamicDWConv(embed_dim , 3, 1, embed_dim)
        self.dyd= DynamicDWConv(embed_dim // 2, 3, 1, embed_dim // 2)

        # saliency
        ######################################################################
        self.t = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.sal_model = TranSalNet()
        self.sal_model.load_state_dict(torch.load(r'all_code/TranSalNet/pretrained_models/TranSalNet_Res.pth'))

        for param in self.sal_model.parameters():
            param.requires_grad = False
        self.sal_model.eval()

        # for param in self.sal_model.parameters():
        #     param.requires_grad = True
        # self.sal_model.train()

        # fc
        ######################################################################
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )
    
    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x


    def forward(self, x, x_saliency):
        x_saliency = x_saliency.type(torch.FloatTensor).cuda()
        x_sal = self.sal_model(x_saliency) # torch.Size([5, 1, 288, 384])
        
        restored_saliency = []
        for i in range(x_sal.shape[0]):  # 遍历批次中的每个图像
            pred = x_sal[i, 0, :, :].cpu().detach().numpy()  # 取出单个显著性图并转换为 numpy 数组
            restored_sal = self.postprocess_img(pred, x[i])  # 恢复大小
            restored_sal = np.expand_dims(restored_sal, axis=0)  # 添加一个通道维度
            restored_sal = np.repeat(restored_sal, 3, axis=0)  # 复制到 3 个通道
            restored_saliency.append(restored_sal)
        x_saliency = torch.tensor(np.array(restored_saliency), dtype=torch.float32).to(x.device)
        
        # saliency way1 vit输入进行融合
        ######################################################################
        # x = self.t * x_saliency + (1 - self.t) * x
        # print('t:', self.t)

        # salient way2 vit输出进行融合加入sigmoid
        ######################################################################
        _x_sal = self.vit(x_saliency)
        x_sal = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
        self.save_output.outputs.clear()

        _x = self.vit(x)
        x = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
        self.save_output.outputs.clear()

        # salient way2 vit输出进行融合加入sigmoid
        ######################################################################
        alpha = torch.sigmoid(self.t)
        x = alpha * x + (1 - alpha) * x_sal
        print('t:', self.t)

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block1(x_vit)
        x_vit1 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        # 方法3_1: res+res+dckg
        ######################################################################
        x_res1 = self.resblock1(x)
        x_res1 = self.dyd_0(x_res1)

        x = torch.cat((x_vit1, x_res1), dim=1) 
        x = self.catconv1(x) # torch.Size([12, 768, 28, 28])
  
        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block2(x_vit)
        x_vit2 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        # 方法3_1: res+res+dckg
        ######################################################################
        x_res2 = self.resblock2(x)
        x_res2 = self.dyd(x_res2)

        x = torch.cat((x_vit2, x_res2), dim=1)
        x = self.catconv2(x) # torch.Size([2, 384, 28, 28])
        
        # fc
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
 
    def postprocess_img(self, pred, org):
        pred = np.array(pred)
        shape_r = org.shape[1]  # 获取原图像的高度
        shape_c = org.shape[2]  # 获取原图像的宽度
        predictions_shape = pred.shape

        rows_rate = shape_r / predictions_shape[0]
        cols_rate = shape_c / predictions_shape[1]

        if rows_rate > cols_rate:
            new_cols = int(predictions_shape[1] * shape_r / predictions_shape[0])
            pred = cv2.resize(pred, (new_cols, shape_r))
            img = pred[:, (new_cols - shape_c) // 2 : (new_cols - shape_c) // 2 + shape_c]
        else:
            new_rows = int(predictions_shape[0] * shape_c / predictions_shape[1])
            pred = cv2.resize(pred, (shape_c, new_rows))
            img = pred[(new_rows - shape_r) // 2 : (new_rows - shape_r) // 2 + shape_r, :]

        return img
