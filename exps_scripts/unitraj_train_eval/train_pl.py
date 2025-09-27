
import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from utils_datasets_traj import build_dataset
from utils.utils_train_traj import set_seed, find_latest_checkpoint
from utils.path_manager import path_manager

from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="config")
def train(cfg):
    path_manager.resolve_config_paths(cfg)
    set_seed(cfg.seed)  # Set seed
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    
    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)
    
    model = build_model(cfg)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)
    
    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace it with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        # dirpath=f'./TrajAttr_ckpt/{cfg.exp_name}',
        dirpath=str(path_manager.get_weights_path("TrajAttr_ckpt", cfg.exp_name)),
    
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="TrajAttr", name=cfg.exp_name, id=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )
    
    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_root = path_manager.get_weights_path("TrajAttr_ckpt", cfg.exp_name)
        search_pattern = os.path.join(str(search_root), "**", "*.ckpt")
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
