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
        self.catconv3 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # two branch
        self.cattxt1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0) # txt stage1
        self.cattxt2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # txt stage2

        # output_hclr
        ######################################################################
        self.hclr = HCLR() 
        # output_dckg
        self.dyd_conv = nn.Conv2d(3, embed_dim // 2, 3, 1, 1)
        self.dyd_0= DynamicDWConv(embed_dim , 3, 1, embed_dim)
        self.dyd= DynamicDWConv(embed_dim // 2, 3, 1, embed_dim // 2)
        # output_txt1
        self.txtconv = nn.Conv2d(6, 3, 1, 1, 0)
        # output_txt2
        self.txtconv3_1 = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.txtconv1_1 = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0) # stage1
        self.txtconv1_2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # stage2

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
        # texture+rgb
        # 方法1: 6通道降为3通道输入到vit
        ######################################################################
        # x = torch.cat((x, x_texture), dim=1)
        # x = self.txtconv(x)
        # x = (x - x.min()) / (x.max() - x.min()) # 添加层归一化
        # x = (x - 0.5) / 0.5
        # texture+rgb
        # 方法2:将texture下采样并使用3x3特征提取
        ######################################################################
        # x_texture = F.interpolate(x_texture, size=(28, 28), mode='bilinear', align_corners=False)
        # x_texture = self.txtconv3_1(x_texture)
        # x_texture1 = self.txtconv1_1(x_texture)
        # x_texture2 = self.txtconv1_2(x_texture)

        _x = self.vit(x)
        x = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block1(x_vit)
        x_vit1 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        # 方法3_2: dckg+res+res
        ######################################################################
        x_res1 = self.dyd_0(x)
        x_res1 = self.resblock1(x)
        # 方法3_1: res+res+dckg
        ######################################################################
        # x_res1 = self.dyd_0(x_res1)

        x = torch.cat((x_vit1, x_res1), dim=1) 
        x = self.catconv1(x) # torch.Size([12, 768, 28, 28])
        
        # texture+rgb
        # 方法2:将texture下采样并使用3x3特征提取
        ######################################################################
        # x = torch.cat((x, x_texture1), dim=1)
        # x = self.cattxt1(x)
  
        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block2(x_vit)
        x_vit2 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        # 方法3_2: dckg+res+res
        ######################################################################
        x_res2 = self.dyd(x)
        x_res2 = self.resblock2(x)
        # 方法3_1: res+res+dckg
        ######################################################################
        # x_res2 = self.dyd(x_res2)

        x = torch.cat((x_vit2, x_res2), dim=1)
        x = self.catconv2(x) # torch.Size([2, 384, 28, 28])
        
        # texture+rgb
        # 方法2:将texture下采样并使用3x3特征提取
        ######################################################################
        # x = torch.cat((x, x_texture2), dim=1)
        # x = self.cattxt2(x)

        # texture attention
        #方法一:使用完整的上下采样流程
        ######################################################################
        # x_texture = self.hclr(x_texture)# torch.Size([2, 384, 224, 224])
        # x_texture = F.interpolate(x_texture, size=(28, 28), mode='bilinear', align_corners=False)
        # x = torch.cat((x, x_texture), dim=1)
        # x = self.catconv3(x) 
        # 方法二:使用完整上下采样流程中间使用3x3+1x1替换feature extraction block
        ######################################################################
        # x_texture = self.hclr(x_texture)
        # x_texture = F.interpolate(x_texture, size=(28, 28), mode='bilinear', align_corners=False)
        # x = x * x_texture
        # 方法三：使用UVEB中DCKG注意力模块
        ######################################################################
        # x_texture = F.interpolate(x_texture, size=(28, 28), mode='bilinear', align_corners=False) #torch.Size([2, 384, 28, 28])
        # x_texture = self.dyd_conv(x_texture) # 3x3
        # x_texture_attention = self.dyd(x_texture)
        # x_texture = x_texture * x_texture_attention
        # x = torch.cat((x, x_texture), dim=1)
        # x = self.catconv3(x) 

        # fc
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score

# import torch
# import torch.nn as nn
# import timm

# from timm.models.vision_transformer import Block
# from timm.models.resnet import BasicBlock,Bottleneck
# # from models.swin import SwinTransformer
# from torch import nn
# from einops import rearrange
# from models.resnet import ResBlockGroup
# from models.hclr import HCLR
# from models.hclr import Attention
# from models.newDyD import DynamicDWConv
# import torch.nn.functional as F

# class TABlock(nn.Module):
#     def __init__(self, dim, drop=0.1):
#         super().__init__()
#         self.c_q = nn.Linear(dim, dim)
#         self.c_k = nn.Linear(dim, dim)
#         self.c_v = nn.Linear(dim, dim)
#         self.norm_fact = dim ** -0.5
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj_drop = nn.Dropout(drop)

#     def forward(self, x):
#         _x = x
#         B, C, N = x.shape
#         q = self.c_q(x)
#         k = self.c_k(x)
#         v = self.c_v(x)

#         attn = q @ k.transpose(-2, -1) * self.norm_fact
#         attn = self.softmax(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, C, N)
#         x = self.proj_drop(x)
#         x = x + _x
#         return x


# class SaveOutput:
#     def __init__(self):
#         self.outputs = []
    
#     def __call__(self, module, module_in, module_out):
#         self.outputs.append(module_out)
    
#     def clear(self):
#         self.outputs = []


# class MANIQA(nn.Module):
#     def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
#                     depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
#                     img_size=224, num_tab=2, scale=0.8, **kwargs):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.input_size = img_size // patch_size
#         self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
#         self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
#         # texture
#         ######################################################################
#         self.resnet50 =  timm.create_model('resnet50',pretrained=True)
#         self.save_output = SaveOutput()
#         hook_handles = []
#         for layer in self.vit.modules():
#             if isinstance(layer, Block):
#                 handle = layer.register_forward_hook(self.save_output)
#                 hook_handles.append(handle)
#         for layer in self.resnet50.modules():
#             if isinstance(layer, Bottleneck):
#                 handle = layer.register_forward_hook(self.save_output)
#                 hook_handles.append(handle)

#         # tab1
#         ######################################################################
#         self.tablock1 = nn.ModuleList()
#         for i in range(num_tab):
#             tab = TABlock(self.input_size ** 2)
#             self.tablock1.append(tab)

#         self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
#         # vit block1
#         ######################################################################
#         self.block1 = Block(
#             dim=embed_dim,
#             num_heads=num_heads[0],
#             mlp_ratio=dim_mlp / embed_dim,
#             drop=drop,
#         )
#         # resblock1
#         ######################################################################
#         self.resblock1 = ResBlockGroup(embed_dim, num_blocks=2)

#         # tab2
#         ######################################################################
#         self.tablock2 = nn.ModuleList()
#         for i in range(num_tab):
#             tab = TABlock(self.input_size ** 2)
#             self.tablock2.append(tab)

#         self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
#         # vit block2
#         ######################################################################
#         self.block2 = Block(
#             dim=embed_dim // 2,
#             num_heads=num_heads[1],
#             mlp_ratio=dim_mlp / (embed_dim // 2),
#             drop=drop,
#         )     
#         # resblock2
#         ######################################################################
#         self.resblock2 = ResBlockGroup(embed_dim // 2, num_blocks=2)
        
#         # reduce dim
#         ######################################################################
#         self.catconv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0) # stage1
#         self.catconv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # stage2
#         self.catconv3 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # two branch
#         self.cattxt1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0) # txt stage1
#         self.cattxt2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # txt stage2
        
#         self.txt1 = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0) # txt1
#         self.txt2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # txt2   
#         self.txtconv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0) # txt and main stage1
#         self.txtconv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0) # txt and main stage2 

#         # output_hclr
#         ######################################################################
#         self.hclr = HCLR() 
#         # output_dckg
#         ######################################################################
#         self.dyd_conv = nn.Conv2d(3, embed_dim // 2, 3, 1, 1)
#         self.dyd_0= DynamicDWConv(embed_dim , 3, 1, embed_dim)
#         self.dyd= DynamicDWConv(embed_dim // 2, 3, 1, embed_dim // 2)

#         # fc
#         ######################################################################
#         self.fc_score = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(embed_dim // 2, num_outputs),
#             nn.ReLU()
#         )
#         self.fc_weight = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(embed_dim // 2, num_outputs),
#             nn.Sigmoid()
#         )
    
#     def extract_feature(self, save_output):
#         x6 = save_output.outputs[6][:, 1:]
#         x7 = save_output.outputs[7][:, 1:]
#         x8 = save_output.outputs[8][:, 1:]
#         x9 = save_output.outputs[9][:, 1:]
#         x = torch.cat((x6, x7, x8, x9), dim=2)
#         return x

#     def get_resnet_feature(self, save_output):
#         feat = torch.cat(
#             (
#                 save_output.outputs[0],
#                 save_output.outputs[1],
#                 save_output.outputs[2]
#             ),
#             dim=1
#         )
#         return feat

#     def forward(self, x, x_texture):
#         # texture 
#         ######################################################################
#         self.resnet50.eval()
#         _ = self.resnet50(x_texture)
#         x_texture = self.get_resnet_feature(self.save_output) # 0,1,2都是[B,256,56,56] # 768 384 28 28
#         x_texture = F.interpolate(x_texture, size=(self.input_size, self.input_size), mode='bilinear', align_corners=True)
#         x_texture_1 = self.txt1(x_texture)
#         x_texture_2 = self.txt2(x_texture) # torch.Size([18, 768, 28, 28]) torch.Size([18, 384, 28, 28])
#         self.save_output.outputs.clear()

#         _x = self.vit(x)
#         x = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
#         self.save_output.outputs.clear()

#         # stage 1
#         x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
#         for tab in self.tablock1:
#             x = tab(x)
#         x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
#         x = self.conv1(x)
#         x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
#         x_vit = self.block1(x_vit)
#         x_vit1 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
#         x_res1 = self.resblock1(x)
#         # 方法3_1: res+res+dckg
#         ######################################################################
#         x_res1 = self.dyd_0(x_res1)
#         x = torch.cat((x_vit1, x_res1), dim=1) 
#         x = self.catconv1(x) # torch.Size([12, 768, 28, 28])
        
#         # 方法3_1: res+res+dckg + texture
#         ######################################################################
#         x = torch.cat((x, x_texture_1), dim=1)
#         x = self.txtconv1(x)

#         # stage2
#         x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
#         for tab in self.tablock2:
#             x = tab(x)
#         x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
#         x = self.conv2(x)
#         x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
#         x_vit = self.block2(x_vit)
#         x_vit2 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
#         x_res2 = self.resblock2(x)
#         # 方法3_1: res+res+dckg
#         ######################################################################
#         x_res2 = self.dyd(x_res2)

#         x = torch.cat((x_vit2, x_res2), dim=1)
#         x = self.catconv2(x) # torch.Size([2, 384, 28, 28])

#         # 方法3_1: res+res+dckg + texture
#         ######################################################################
#         x = torch.cat((x, x_texture_2), dim=1)  
#         x = self.txtconv2(x)

#         # fc
#         x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
#         score = torch.tensor([]).cuda()
#         for i in range(x.shape[0]):
#             f = self.fc_score(x[i])
#             w = self.fc_weight(x[i])
#             _s = torch.sum(f * w) / torch.sum(w)
#             score = torch.cat((score, _s.unsqueeze(0)), 0)
#         return score
