"""
加入texture作为query的自连接
保留maniqa框架
"""
import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
# from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
from models.resnet import ResBlockGroup
from models.newDyD import DynamicDWConv #dckg
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
            tab = TEABlock(self.input_size ** 2)
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

        # texture
        ######################################################################
        self.ta = TA(3072)
        self.t_conv1 = nn.Conv2d(3072, 3072, 1, 1, 0)
        self.t_conv2 = nn.Conv2d(3072, 3072, 1, 1, 0)

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
    
    def forward(self, x, x_texture):
        # texture attention
        ######################################################################
        x_ta = self.ta(x_texture)
        x_texture1 = self.t_conv1(x_ta) # torch.Size([1, 3072, 28, 28])
        x_texture2 = self.t_conv2(x_ta) # torch.Size([1, 3072, 28, 28])

        # vit
        ######################################################################
        _x = self.vit(x)
        x = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        x_texture1 = rearrange(x_texture1, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        x_texture2 = rearrange(x_texture2, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        ######################################################################
        x = self.tablock1[0](x, x_texture1) 
        x = self.tablock1[1](x, x_texture2)
        ######################################################################
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        ######################################################################
        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block1(x_vit)
        x_vit1 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        x_res1 = self.resblock1(x)
        x_res1 = self.dyd_0(x_res1)

        x = torch.cat((x_vit1, x_res1), dim=1) 
        x = self.catconv1(x) # torch.Size([12, 768, 28, 28])
        ######################################################################
  
        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        ######################################################################
        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block2(x_vit)
        x_vit2 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        x_res2 = self.resblock2(x)
        x_res2 = self.dyd(x_res2)

        x = torch.cat((x_vit2, x_res2), dim=1)
        x = self.catconv2(x) # torch.Size([2, 384, 28, 28])
        ######################################################################
    
        # fc
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score

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

class TEABlock(nn.Module): # 保留原MANIQA TABlock自连接使用texture作为query
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.c_b = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x_qkv, x_b):
        _x = x_qkv # torch.Size([20, 3072, 784])
        B, C, N = x_qkv.shape
        q = self.c_q(x_qkv) #
        k = self.c_k(x_qkv)  # torch.Size([20, 3072, 784])
        v = self.c_v(x_qkv)

        b = self.c_b(x_b) # torch.Size([14, 3072, 3072])
        b = b.mean(dim=-1, keepdim=True)
        b = b.repeat(1, 1, C)

        attn = (q @ k.transpose(-2, -1) + b) * self.norm_fact
        attn = self.softmax(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x

class TA(nn.Module): # texture attention
    def __init__(self, channels):
        super(TA, self).__init__()
        self.texture_attention = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d((28, 28))
    def forward(self, texture_image):
        x_ta = self.pool(texture_image) # torch.Size([28, 3, 28, 28])
        x_ta = self.texture_attention(x_ta)
        return x_ta