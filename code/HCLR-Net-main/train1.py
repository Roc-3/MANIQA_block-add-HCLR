import pytorch_lightning as pl
import os
import kornia.metrics.psnr as PSNR
import kornia.metrics.ssim as SSIM
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from loss.L1 import L1_loss
from loss.Perceptual_our import PerceptualLoss
from loss.UCR import UnContrastLoss
from argparse import Namespace
from dataloader3 import Haze4kdataset, Val4kdataset
from pytorch_lightning import seed_everything
from network import Network
from pytorch_lightning.loggers import TensorBoardLogger

seed = 42  # Global seed set to 42
seed_everything(seed)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logger = TensorBoardLogger('tb_logs', name='UCR')

class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams

        # train/val/test datasets
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.test_datasets = self.params.test_datasets
        self.test_batchsize = self.params.test_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs

        # Train setting
        self.initlr = self.params.initlr  # initial learning
        self.weight_decay = self.params.weight_decay  # optimizers weight decay
        self.crop_size = self.params.crop_size  # random crop size
        self.num_workers = self.params.num_workers

        # loss_function
        self.loss_L1 = L1_loss().cuda()
        self.loss_Pe = PerceptualLoss().cuda()
        self.UCR = UnContrastLoss().cuda()
        self.model = Network()

    def forward(self, x):
        y_lable = self.model(x)
        return y_lable
    def forward1(self, x):
        y = self.model(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.d, betas=[0.9, 0.999],
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.initlr, max_lr=1.5 * self.initlr,
                                                      cycle_momentum=False)

        return [optimizer], [scheduler]


    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward1(x)
        loss = self.loss_L1(y_hat, y) + 0.2 * self.loss_Pe(y_hat, y)
        ssim = SSIM(y_hat, y, 5).mean().item()
        psnr = PSNR(y_hat, y, 1).item()

        self.log('val_loss', loss)
        self.log('psnr', psnr)
        self.log('ssim', ssim)
        self.trainer.checkpoint_callback.best_model_score

        return {'val_loss': loss, 'psnr': psnr, 'ssim': ssim}

    def train_dataloader(self):
        train_set = Haze4kdataset(self.train_datasets, train=True, size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True,
                                                   num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_set = Val4kdataset(self.validation_datasets, train=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False,
                                                 num_workers=self.num_workers)

        return val_loader


def main():
    RESUME = False
    resume_checkpoint_path = r''

    args = {
        'epochs': 500,
        # datasetsw
        'train_datasets': r'Datasets/train',
        'test_datasets': None,
        'val_datasets': r'Datasets/val',
        # bs
        'train_bs': 16,
        # 'train_bs':4,
        'test_bs': 1,
        'val_bs': 8,
        # 'initlr':0.0002,
        'initlr': 0.0001,
        'weight_decay': 0.001,
        'crop_size': 256,
        'num_workers': 0,
        # Net
        'model_blocks': 5,
        'chns': 64
    }

    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor='psnr',
        filename='epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}-train_loss{train_loss:.3f}-val_loss{val_loss:.3f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=3,
        mode="max"
    )

    if RESUME:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=[0],
            resume_from_checkpoint=resume_checkpoint_path,
            logger=logger,
            strategy='dp',
            precision=16,
            callbacks=[checkpoint_callback],

        )
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=[0],
            logger=logger,
            strategy='dp',
            precision=16,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(model)


if __name__ == '__main__':
    main()
