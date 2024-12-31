
class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):

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
