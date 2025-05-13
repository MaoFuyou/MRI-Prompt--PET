
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from scripts.vq_gan_3d.model.vqgan import VQGAN
from scripts.train.callbacks import ImageLogger, VideoLogger
from omegaconf import OmegaConf, DictConfig, open_dict

from scripts.train.get_dataset import get_dataset

@pl.utilities.rank_zero_only
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sampler=sampler,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        shuffle=False
    )

    bs, base_lr, ngpu, accumulate = (
        cfg.model.batch_size,
        cfg.model.lr,
        cfg.model.gpus,
        cfg.model.accumulate_grad_batches,
    )

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu / 8.0) * (bs / 4.0) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix
        )
    print("¾nf`: {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu / 8, bs / 4, base_lr))

    model = VQGAN(cfg)
    model.cuda()

    callbacks = []
    callbacks.append(ModelCheckpoint(
        monitor='val/recon_loss',
        save_top_k=3,
        mode='min',
        filename='latest_checkpoint'
    ))
    callbacks.append(ModelCheckpoint(
        every_n_train_steps=3000,
        save_top_k=-1,
        filename='{epoch}-{step}-{train/recon_loss:.2f}'
    ))
    callbacks.append(ModelCheckpoint(
        every_n_train_steps=10000,
        save_top_k=-1,
        filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'
    ))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))


    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ''
        ckpt_file = ''
        version_id_used = 0
        for folder in os.listdir(base_dir):
            try:
                version_id = int(folder.split('_')[1])
            except Exception:
                continue
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if log_folder:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            if os.path.exists(ckpt_folder):
                for fn in os.listdir(ckpt_folder):
                    if fn == 'latest_checkpoint.ckpt':
                        ckpt_file = 'latest_checkpoint_prev.ckpt'
                        os.rename(os.path.join(ckpt_folder, fn),
                                  os.path.join(ckpt_folder, ckpt_file))
                if ckpt_file:
                    cfg.model.resume_from_checkpoint = os.path.join(
                        ckpt_folder, ckpt_file)
                    print(f'learning rate¹ {cfg.model.resume_from_checkpoint} 