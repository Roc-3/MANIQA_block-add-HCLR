import torch
import torchvision
from pathlib import Path
from train1 import CoolSystem as UIE_cp
import sys
import os
import argparse
import platform
from core.Losses import (Metrics, PSNR, SSIM, NegMetric)
from core.utils.general import colorstr, init_seeds, yaml_save
from core.monitor import Monitor
from PIL import Image
from argparse import Namespace
from core.Datasets.creator import SingleFolder
import thop

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # get root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from core.Datasets.builder import cache_dataset, build_dataloader

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
LOG = Monitor('./bbb')

def eval(model, dataloader, save_dir, metrics, device):
    model.eval()
    metrics.clear()
    with torch.no_grad():
        for _, im in enumerate(dataloader):
            if 'gt' in im:
                input , gt = im['raw'].to(device), im['gt'].to(device)
            else:
                input = im['raw'].to(device)
            pred = model(input).clamp(0., 1.)
            if 'gt' in im:
                metrics(pred, gt)
                image_array = torch.cat([input.to('cpu'), gt.to('cpu'), pred.to('cpu')], dim=2).squeeze().permute(1, 2, 0)
            else:
                metrics(pred, pred)
                image_array = torch.cat([pred.to('cpu')], dim=2).squeeze().permute(1, 2, 0)
            print(_)
            im = Image.fromarray((image_array * 255).clamp(0, 255).add_(0.5).numpy().astype('uint8'))
            im.save("{}.png".format(os.path.join(save_dir, str(_))))
            LOG.metricsWriter(_, metrics.back())
        metric = metrics.output(len(dataloader))
    return  metric


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=250, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=5, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

from torchvision import transforms


def apply_transforms(examples):
    examples["raw"] = [jitter(image.convert("RGB")) for image in examples["raw"]]
    
    if "gt" in examples:
        examples["gt"] = [jitter(image.convert("RGB")) for image in examples["gt"]]

    return examples


jitter = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def main(hyp, opt, device):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, evolve, data, cfg, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.evolve, opt.data, opt.cfg, \
        opt.noval, opt.nosave, opt.workers, opt.freeze
    args = {
        'epochs': 100,
        # datasetsw
        'train_datasets': r'/home/hikari/code/datasets/UIEB/UIEB_end/train',
        'test_datasets': r'/home/hikari/code/datasets/UIEB/UIEB_end/test',
        'val_datasets': r'/home/hikari/code/datasets/UIEB/UIEB_end/val',
        # bs
        'train_bs': 16,
        # 'train_bs':4,
        'test_bs': 1,
        'val_bs': 1,
        # 'initlr':0.0002,
        'initlr': 0.0003,
        'weight_decay': 0.001,
        'crop_size': 256,
        'num_workers': 0,
        # Net
        'model_blocks': 5,
        'chns': 64
    }
    hparams = Namespace(**args)
    model = UIE_cp(hparams).to(device)
    model.load_state_dict(torch.load(opt.weights)["state_dict"])
    dataset = cache_dataset('new', data, apply_transforms)


    val_loader = build_dataloader(dataset['val'](), 1, workers)
    test_loader = build_dataloader(dataset['test'](), 1, workers)
    ts = SingleFolder('haze', auto_save_disk=False, split='test')
    ts.load('/share/zhangdan2013/code/datasets/test_images/origin')
    ts.set_transform(apply_transforms)
    singer_test = build_dataloader(ts(), 1, workers)

    
    metrics = Metrics()
    metrics.add([PSNR(), SSIM(), NegMetric()])
    im = torch.empty((1, 3, 256, 256), device=device)
    print(sum(x.numel() for x in model.parameters()))
    print(thop.profile(model, inputs=(im, ), verbose=False)[0] / 1E9)

    metric = eval(model, val_loader, save_dir / 'val' , metrics, device)
    print('avg Val90:{}'.format(metric))

    metric = eval(model, test_loader, save_dir / 'origin', metrics, device)
    print('avg Test60:{}'.format(metric))

if __name__ == '__main__':
    opt = parse_opt()
    opt.save_dir = opt.project
    init_seeds(opt.seed)
    main(opt.hyp, opt, torch.device('cuda:'+ opt.device))