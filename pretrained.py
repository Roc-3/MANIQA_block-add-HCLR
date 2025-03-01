import torch
import os
import numpy as np
import cv2

from pretrained_maniqa import MANIQA
from config import Config

from PIL import Image
from torchvision import transforms

from data.livec.livec import LIVEC
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from utils.slic.slic_func import SLIC
import matplotlib.pyplot as plt
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = Config({
    # model
    "patch_size": 8,
    "img_size": 224,
    "embed_dim": 768,
    "dim_mlp": 768,
    "num_heads": [4, 4],
    "window_size": 4,
    "depths": [2, 2],
    "num_outputs": 1,
    "num_tab": 2,
    "scale": 0.8,
    
    # load & save checkpoint
    "ckpt_path": "all_save_dataset/output_livec_best/models/livec/epoch37.pt"
})

net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
    patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
    depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

net.load_state_dict(torch.load(config.ckpt_path), strict=False)
net = net.cuda()

# 记录网络层输入和输出
def register_hooks(model):
    layer_inputs_outputs = []

    # 钩子函数
    def hook_fn(module, input, output):
        layer_name = module.__class__.__name__
        layer_input = input[0].shape  # 获取输入的形状
        layer_output = output.shape  # 获取输出的形状

        layer_inputs_outputs.append({
            "Layer": layer_name,
            "Input shape": layer_input,
            "Output shape": layer_output
        })

    # 注册钩子
    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)  # 注册钩子
        hooks.append(hook)

    return hooks, layer_inputs_outputs

def save_layer_info(layer_inputs_outputs, filename="layer_inputs.txt"):
    with open(filename, "w") as f:
        for layer_info in layer_inputs_outputs:
            f.write(f"Layer: {layer_info['Layer']}\n")
            f.write(f"Input shape: {layer_info['Input shape']}\n")
            f.write(f"Output shape: {layer_info['Output shape']}\n\n")

def gaussian_filter(image, sigma):
    """
    对图像应用高斯滤波器
    """
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

def structure_texture_decomposition(image, sigma):
    """
    使用高斯滤波器进行结构-纹理分解
    :param image: 输入图像
    :param sigma: 高斯滤波器的标准差
    :return: 结构图像和纹理图像
    """
    # 计算结构图像（高斯模糊）
    structure_image = gaussian_filter(image, sigma)
    
    # 计算纹理图像（原图减去结构图像）
    texture_image = image - structure_image
    
    return structure_image, texture_image

# 读取原始评分数据的最小值和最大值
def get_min_max_scores(label_file):
    scores = []
    with open(label_file, 'r') as f:
        for line in f:
            _, score = line.split()
            scores.append(float(score))
    return min(scores), max(scores)


def generate_gradcam_heatmap(model, input_image, d_img_texture, d_img_slic, target_layer):
    """
    生成Grad-CAM热力图
    :param model: 训练好的模型
    :param input_image: 输入图像
    :param target_layer: 目标卷积层
    :return: 热力图
    """
    model.eval()

    # 初始化梯度变量
    gradients = None

    # 钩子函数：保存梯度
    def save_gradient(grad):
        nonlocal gradients
        gradients = grad  # 将梯度保存到梯度变量

    # 获取指定层的输出
    layer_output = None
    def hook_fn(module, input, output):
        nonlocal layer_output
        layer_output = output
        output.register_hook(save_gradient)  # 注册梯度钩子

    target_layer.register_forward_hook(hook_fn)

    # 计算模型的输出
    output = model(input_image, d_img_texture, d_img_slic)

    # 获取目标评分：质量评估任务，直接取输出的评分
    target_score = output[0]  # 直接使用输出的评分（标量）

    # 计算梯度
    model.zero_grad()
    target_score.backward(retain_graph=True)  # 反向传播以计算梯度

    # 检查梯度是否为零
    if gradients is None or torch.all(gradients == 0):
        print("Warning: Gradients are zero.")

    # 获取该层的梯度
    gradients = gradients[0]  # 获取梯度

    # 如果梯度维度不足4，进行调整
    if gradients.dim() == 3:  # 假设只有 [channels, height, width]
        gradients = gradients.unsqueeze(0)  # 添加 batch 维度

    # 平均池化梯度
    weights = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    
    # 计算Grad-CAM的加权和
    gradcam_output = torch.sum(weights * layer_output, dim=1, keepdim=True)
    # gradcam_output = F.relu(gradcam_output)
    
    # 归一化热力图
    gradcam_output = gradcam_output.squeeze().cpu().detach().numpy()
    gradcam_output = cv2.resize(gradcam_output, (input_image.shape[2], input_image.shape[3]))
    gradcam_output -= np.min(gradcam_output)
    gradcam_output /= np.max(gradcam_output)
    
    return gradcam_output


"""
主函数设置
"""
label_file = 'data/livec/livec_label.txt'
min_score, max_score = get_min_max_scores(label_file)

# 加载并预处理图像
d_img = cv2.imread('pretrained/fisher.png', cv2.IMREAD_COLOR)

# vit
d_img_vit = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
d_img_vit = cv2.cvtColor(d_img_vit, cv2.COLOR_BGR2RGB)
d_img_vit = np.array(d_img_vit).astype('float32') / 255
d_img_vit = np.transpose(d_img_vit, (2, 0, 1))
normalize = Normalize(0.5, 0.5)
d_img_vit = normalize(d_img_vit) # (3, 224, 224)

# texture
d_img_gray = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
_, d_img_texture = structure_texture_decomposition(d_img_gray, sigma=2.0)
d_img_texture = d_img_texture.astype('float32') / 255
d_img_texture = np.expand_dims(d_img_texture, axis=0)
d_img_texture = np.repeat(d_img_texture, 3, axis=0) # (3, 500, 500)
d_img_texture = np.transpose(d_img_texture, (1, 2, 0)) # (500, 500, 3)
d_img_texture = cv2.resize(d_img_texture, (224, 224), interpolation=cv2.INTER_CUBIC) # (224, 224, 3)
d_img_texture = torch.tensor(d_img_texture, dtype=torch.float32)
d_img_texture = np.transpose(d_img_texture, (2, 0, 1))

texture_image_path = 'pretrained/texture_image.png'
cv2.imwrite(texture_image_path, d_img_texture.cpu().numpy().transpose(1, 2, 0) * 255)

# slic
d_img_slic = cv2.resize(d_img, (500, 500), interpolation=cv2.INTER_CUBIC)
d_img_slic = np.array(d_img_slic).astype('uint8') # hwc
slic_args = {
    'image_n_nodes': 140,
    'patch_n_nodes': 600,
    'region_size': 40,
    'ruler': 10.0,
    'iterate': 10
}
slic_class = SLIC(img=d_img_slic, args=slic_args)
d_img_slic = slic_class.slic_function(save_path='pretrained/slic_result.npy', visualize_path='pretrained/slic_visualization.png')
d_img_slic = d_img_slic.astype('float32') / 255 # (image_n_nodes, patch_n_nodes, 3)

# all
slic_image_path_all = 'pretrained/slic_image_all.png'
cv2.imwrite(slic_image_path_all, d_img_slic * 255)

# to tensor
d_img_vit = torch.tensor(d_img_vit).unsqueeze(0).cuda()
d_img_texture = torch.tensor(d_img_texture).unsqueeze(0).cuda()
d_img_slic = torch.tensor(d_img_slic).unsqueeze(0).cuda()

# 注册钩子
hooks, layer_inputs_outputs = register_hooks(net)

net.eval()
with torch.no_grad():
    output = net(d_img_vit, d_img_texture, d_img_slic)
    normalized_score = output.item()

# 记录网络层的输入和输出到文件
save_layer_info(layer_inputs_outputs, "layer_inputs.txt")

# 获取Grad-CAM热力图
target_layer = net.fusionconv  # 请根据实际层名指定你想要查看的层
if target_layer is None:
    raise ValueError("Please specify the target layer")
gradcam_heatmap = generate_gradcam_heatmap(net, d_img_vit, d_img_texture, d_img_slic, target_layer)

# 保存并显示Grad-CAM热力图
gradcam_image_path = 'gradcam_heatmap.png'
plt.imshow(gradcam_heatmap, cmap='jet', alpha=0.5)
plt.savefig(gradcam_image_path)
plt.show()

original_score = normalized_score * (max_score - min_score) + min_score
print('Image quality score:', original_score)

# 清理钩子
for hook in hooks:
    hook.remove()