import os
import torch
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import align_file_with_reference
from data.livec.livec_test import LIVEC
from tqdm import tqdm

from models.maniqa import MANIQA

from scipy.stats import spearmanr, pearsonr

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_scores_from_file(file_path):
    """ 读取评分文件，返回字典 {image_name: score} """
    scores_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, score = parts
                scores_dict[image_name] = float(score)
    return scores_dict

def normalize_scores(score_dict):
    """ 归一化分数到 [0, 1] """
    scores = np.array(list(score_dict.values()))
    min_val, max_val = np.min(scores), np.max(scores)
    norm_scores = (scores - min_val) / (max_val - min_val)
    return {k: v for k, v in zip(score_dict.keys(), norm_scores)}

def compute_correlation(gt_file, pred_file):
    """ 计算 SRCC 和 PLCC """
    # 读取真实分数和预测分数
    gt_scores = read_scores_from_file(gt_file)
    pred_scores = read_scores_from_file(pred_file)

    # 归一化真实分数
    gt_scores = normalize_scores(gt_scores)

    # 对齐数据（确保两者的图片顺序一致）
    common_keys = list(set(gt_scores.keys()) & set(pred_scores.keys()))
    common_keys.sort()  # 排序以确保对齐

    gt_values = np.array([gt_scores[k] for k in common_keys])
    pred_values = np.array([pred_scores[k] for k in common_keys])

    # 计算 SRCC 和 PLCC
    srcc, _ = spearmanr(gt_values, pred_values)
    plcc, _ = pearsonr(gt_values, pred_values)

    print(f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
    return srcc, plcc

def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    x_t = data['d_img_texture'].cuda()
                    x_s = data['d_img_slic'].cuda()
                    # x_d = five_point_crop(i, d_img=x_d, config=config)
                    pred += net(x_d, x_t, x_s)

                pred /= config.num_avg_val
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
            for i in range(len(name_list)):
                f.write(name_list[i] + ' ' + str(pred_list[i]) + '\n')
            print(len(name_list))
        f.close()


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "db_name": "tid2013",
        "test_dis_path": "../all_dataset/LIVEC/Images/",
        
        # optimization
        "batch_size": 1,
        "num_avg_val": 1,
        "crop_size": 224,

        # device
        "num_workers": 8,

        # model
        "embed_dim": 768,

        # load & save checkpoint
        "valid": "./output/valid",
        "valid_path": "./output/valid/inference_valid",
        "model_path": "./all_save_dataset/output_tid2013/models/tid2013/epoch130.pt"
    })

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)
    
    # data load
    test_dataset = LIVEC(dis_path=config.test_dis_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    net = MANIQA(embed_dim=config.embed_dim)
    net.load_state_dict(torch.load(config.model_path))
    net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    
    # get srcc plcc
    gt_file_path = "data/livec/livec_label.txt"
    align_file_with_reference(config.valid_path + '/output.txt', gt_file_path)
    pred_file_path = "output.txt"
    compute_correlation(gt_file_path, pred_file_path)