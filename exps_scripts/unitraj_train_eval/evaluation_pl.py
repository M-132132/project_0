
"""
pytorch_lighting 的 evaluation
"""
import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from utils_datasets_traj import build_dataset
from utils.utils_train_traj import set_seed
import hydra
from omegaconf import OmegaConf
from utils.path_manager import path_manager


@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="config")
# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluation(cfg):
    path_manager.resolve_config_paths(cfg)
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)

    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    trainer = pl.Trainer(
        inference_mode=True,
        logger=None if cfg.debug else WandbLogger(project="TrajAttr", name=cfg.exp_name),
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
    )

    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    evaluation()
