# hclr 
import torch
import torch.nn as nn
import torch.nn.functional as F

class HCLR(nn.Module):
    def __init__(self, input_nc=3, ngf=64, use_dropout=False, padding_type='reflect'):
        super(HCLR, self).__init__()
        ###### downsample
        self.Pad2d1 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(input_nc, ngf, kernel_size=3),
                                    nn.GELU())
        self.block1 = Attention(ngf)
        self.down1 = nn.Sequential(nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1),
                                   nn.GELU())
        self.block2 = Attention(ngf)
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.GELU())
        self.block3 = Attention(ngf*2)
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.GELU())
        ###### blocks
        # 方法一方法二对应的block更改
        # self.block = Block(default_conv, ngf * 4)
        self.block = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1),
            nn.GELU()
        )

        ###### upsample
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU())
        self.block4 = Attention(ngf*2)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU())
        self.block5 = Attention(ngf)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.GELU())
        self.block6 = Attention(ngf)
        self.Pad2d2 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(ngf, 384, kernel_size=3),
                                    nn.Tanh()) # 最后输出的3通道更改为384通道
        # HCLR detail repair branch
        # self.start_conv = default_conv(in_channels=3, out_channels=256, kernel_size=3, bias=True)
        # self.Residual_block = residual_block(in_channels=256, out_channels=256, kernel_size=3)
        # self.final_conv = default_conv(in_channels=256, out_channels=3, kernel_size=3, bias=True)
    def forward(self, input):

        x = self.Pad2d1(input)
        x = self.block1(x)
        x_down1 = self.down1(x)
        x_down1 = self.block2(x_down1)
        x_down2 = self.down2(x_down1)
        x_down2 = self.block3(x_down2)
        x_down3 = self.down3(x_down2)
        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)
        x_up1 = self.up1(x6)
        x_up1 = self.block4(x_up1)
        x_up1 = F.interpolate(x_up1, x_down2.size()[2:], mode='bilinear', align_corners=True)
        add1 = x_down2 + x_up1
        x_up2 = self.up2(add1)
        x_up2 = self.block5(x_up2)
        x_up2 = F.interpolate(x_up2, x_down1.size()[2:], mode='bilinear', align_corners=True)
        add2 = x_down1 + x_up2
        x_up3 = self.up3(add2)
        x_up3 = self.block6(x_up3)
        x_up3 = F.interpolate(x_up3, x.size()[2:], mode='bilinear', align_corners=True)
        add3 = x + x_up3
        result = self.Pad2d2(add3) # result1 as result

        # HCLR detail repair branch
        # conv = self.start_conv(input)
        # Residual_block1 = self.Residual_block(conv)
        # Residual_block2 = self.Residual_block(Residual_block1)
        # Residual_block3 = self.Residual_block(Residual_block2)
        # Residual_block4 = self.Residual_block(Residual_block3)
        # Residual_block5 = self.Residual_block(Residual_block4)
        # Residual_block5 = conv + Residual_block5
        # result2 = self.final_conv(Residual_block5)
        # result = result1 + result2
        return result

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.max = nn.AdaptiveMaxPool2d((1, None))
        self.conv1x1 = default_conv(dim, dim//2, kernel_size=1, bias=True)
        self.conv3x3 = default_conv(dim//2, dim, kernel_size=3, bias=True)
        self.con3x3 = default_conv(dim, dim, kernel_size=3, bias=True)
        self.GELU = nn.GELU()
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x_h = self.avg(x)
        x_w = self.max(x)
        x_h = torch.squeeze(x_h, 3)
        x_w = torch.squeeze(x_w, 2)
        x_h1 = x_h.unsqueeze(3)
        x_w1 = x_w.unsqueeze(2)
        x_h_w = catcat(x_h, x_w)
        x_h_w = x_h_w.unsqueeze(3)
        x_h_w = self.conv1x1(x_h_w)
        x_h_w = self.GELU(x_h_w)
        x_h_w = torch.squeeze(x_h_w, 3)
        x1, x2 = torch.split(x_h_w, [height, width], 2)
        x1 = x1.unsqueeze(3)
        x2 = x2.unsqueeze(2)
        x1 = self.conv3x3(x1)
        x2 = self.conv3x3(x2)
        mix1 = self.mix1(x_h1, x1)
        mix2 = self.mix2(x_w1, x2)
        x1 = self.con3x3(mix1)
        x2 = self.con3x3(mix2)
        matrix = torch.matmul(x1, x2)
        matrix = torch.sigmoid(matrix)
        final = torch.mul(x, matrix)
        final = x + final
        return final

class Block(nn.Module):
    def __init__(self, conv, dim):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, 3, bias=True)
        self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, 1, bias=True)
        self.act2 = nn.GELU()
        self.conv3 = conv(dim, dim, 3, bias=True)
        self.attention = Attention(dim)
    def forward(self, x):
        res1 = self.act1(self.conv1(x))
        res2 = self.act2(self.conv2(x))
        res = res1 + res2
        res = x + res
        res = self.attention(res)
        res = self.conv3(res)
        res = x + res
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def catcat(inputs1, inputs2):
    return torch.cat((inputs1, inputs2), 2)

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out