import os  # 操作系统接口
import time  # 时间相关功能
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import DataLoader  # PyTorch数据加载器
from tqdm import tqdm  # 进度条工具
import numpy as np  # 数值计算库
from models import build_model  # 模型构建函数
from utils_datasets_traj import build_dataset  # 数据集构建函数
from utils.utils_train_traj import set_seed, find_latest_checkpoint  # 训练工具函数
from utils.logger import Logger  # 日志记录器
import hydra  # Hydra配置管理工具
from omegaconf import OmegaConf  # 配置文件操作工具
from utils.path_manager import path_manager  # 路径管理工具


@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="config")
# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg):
    path_manager.resolve_config_paths(cfg)  # 解析配置文件中的路径
    set_seed(cfg.seed)  # 设置随机种子以确保实验可复现
    OmegaConf.set_struct(cfg, False)  # 允许动态添加配置项
    cfg = OmegaConf.merge(cfg, cfg.method)  # 合并基础配置和方法特定配置
    
    # 创建数据集  （仅仅是创建）
    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)
    
    # 计算批次大小
    if len(cfg.devices) > 1 and not cfg.debug:
        train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices), 1)
        eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)
    else:
        train_batch_size = cfg.method['train_batch_size']
        eval_batch_size = cfg.method['eval_batch_size']
    
    # 创建数据加载器（已经转化为H5数据）
    if len(cfg.devices) > 1 and not cfg.debug:
        # 分布式训练
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
        
        train_loader = DataLoader(
            train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers,
            sampler=train_sampler, collate_fn=train_set.collate_fn, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers,
            sampler=val_sampler, collate_fn=train_set.collate_fn, pin_memory=True
        )
    else:
        # 单GPU或CPU训练
        train_loader = DataLoader(
            train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers,
            shuffle=True, drop_last=False, collate_fn=train_set.collate_fn
        )
        
        val_loader = DataLoader(
            val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers,
            shuffle=False, drop_last=False, collate_fn=train_set.collate_fn
        )
    
    # 创建训练器
    trainer = Trainer(cfg)
    
    # 自动恢复训练
    if cfg.ckpt_path is None and not cfg.debug:
        # search_pattern = os.path.join('../../TrajAttr_ckpt', cfg.exp_name, '**', '*.ckpt')
        search_pattern = os.path.join(str(path_manager.get_weights_path('TrajAttr_ckpt')),
                                      cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)
    
    if cfg.ckpt_path:
        trainer.load_checkpoint(cfg.ckpt_path)
    
    # 开始训练
    trainer.fit(train_loader, val_loader, cfg.method.max_epochs)

    
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config.debug else 'cpu')
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # 设置分布式训练
        self.is_distributed = len(config.devices) > 1 and not config.debug
        if self.is_distributed:
            torch.distributed.init_process_group(backend='nccl')
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.local_rank = 0
        
        # 初始化日志记录器
        if self.local_rank == 0:
            log_dir = path_manager.get_weights_path("TrajAttr_ckpt", "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.logger = Logger(str(log_dir), config.exp_name)
            self.logger.log_config(config)
            self.logger.log_text(f"Training started on device: {self.device}")
        
        # 初始化模型
        self.model = build_model(config)
        self.model.to(self.device)
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        
        # 配置优化器和调度器
        if hasattr(self.model, 'configure_optimizers'):
            if self.is_distributed:
                optimizers, schedulers = self.model.module.configure_optimizers()
            else:
                optimizers, schedulers = self.model.configure_optimizers()
            self.optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
            self.scheduler = schedulers[0] if isinstance(schedulers, list) and len(schedulers) > 0 else None
        else:
            # 默认优化器
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.method.learning_rate)
            self.scheduler = None
        
        # 梯度裁剪
        self.grad_clip_norm = config.method.get('grad_clip_norm', None)

        self.checkpoint_dir = str(path_manager.get_weights_path("TrajAttr_ckpt", config.exp_name))
        if self.local_rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch, metric_value, is_best=False):
        """保存检查点"""
        if self.local_rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新的检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.ckpt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            # best_path = os.path.join(self.checkpoint_dir, f'epoch-{epoch}-{metric_value:.4f}.ckpt')
            best_path = os.path.join(self.checkpoint_dir, f'best_model.ckpt')
            torch.save(checkpoint, best_path)
            self.logger.log_text(f"New best model saved with metric: {metric_value:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            if self.local_rank == 0:
                self.logger.log_text(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)#包含权重
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        if self.local_rank == 0:
            self.logger.log_text(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_one_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        batch_losses = []
        
        if self.local_rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            self._move_batch_to_device(batch)
            
            self.optimizer.zero_grad()
            
            # 前向传播，batch中储存着
            prediction, loss = self.model(batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            if self.local_rank == 0:
                pbar.set_postfix({'loss': loss.item()})
                
                # 每100个batch记录一次
                if batch_idx % 100 == 0:
                    metrics = {
                        'train_batch_loss': loss.item(),
                        'train_lr': self.optimizer.param_groups[0]['lr']
                    }
                    self.logger.log_metrics(metrics, epoch=self.current_epoch, step=self.global_step)
        
        avg_loss = total_loss / num_batches
        
        # 记录epoch级别的训练指标
        if self.local_rank == 0:
            train_metrics = {
                'train_epoch_loss': avg_loss,
                'train_loss_std': np.std(batch_losses)
            }
            self.logger.log_metrics(train_metrics, epoch=self.current_epoch + 1)
        
        return avg_loss
    
    def validate_one_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_metrics = []
        
        # 清空预测字典
        if self.is_distributed:
            self.model.module.pred_dicts = []
        else:
            self.model.pred_dicts = []
        
        with torch.no_grad():
            if self.local_rank == 0:
                pbar = tqdm(val_loader, desc='Validation')
            else:
                pbar = val_loader
            
            for batch in pbar:
                # 将数据移到设备
                self._move_batch_to_device(batch)
                
                # 前向传播
                prediction, loss = self.model(batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 计算官方评估指标
                if self.is_distributed:
                    self.model.module.compute_official_evaluation(batch, prediction)
                    metrics = self.model.module.compute_metrics_and_log(batch, prediction, status='val')
                else:
                    self.model.compute_official_evaluation(batch, prediction)
                    metrics = self.model.compute_metrics_and_log(batch, prediction, status='val')
                
                all_metrics.append(metrics)
        
        avg_loss = total_loss / num_batches
        
        # 聚合指标
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # 计算评估指标
        if self.local_rank == 0:
            if self.config.get('eval_waymo', False):
                if self.is_distributed:
                    pred_dicts = self.model.module.pred_dicts
                else:
                    pred_dicts = self.model.pred_dicts
                
                if len(pred_dicts) > 0:
                    metric_results, result_format_str = self.model.compute_metrics_waymo(
                        pred_dicts) if not self.is_distributed else self.model.module.compute_metrics_waymo(pred_dicts)
                    self.logger.log_text("Waymo Evaluation Results:")
                    self.logger.log_text(metric_results)
                    self.logger.log_text(result_format_str)
                    
                    # 保存预测结果
                    self.logger.save_predictions(pred_dicts, f'waymo_predictions_epoch_{self.current_epoch + 1}.json')
            
            elif self.config.get('eval_nuscenes', False):
                if self.is_distributed:
                    pred_dicts = self.model.module.pred_dicts
                else:
                    pred_dicts = self.model.pred_dicts
                
                if len(pred_dicts) > 0:
                    os.makedirs('submission', exist_ok=True)
                    import json
                    json.dump(pred_dicts, open(os.path.join('submission', "evalai_submission.json"), "w"))
                    metric_results = self.model.compute_metrics_nuscenes(
                        pred_dicts) if not self.is_distributed else self.model.module.compute_metrics_nuscenes(
                        pred_dicts)
                    self.logger.log_text("NuScenes Evaluation Results:")
                    self.logger.log_text(str(metric_results))
                    
                    # 保存预测结果
                    self.logger.save_predictions(pred_dicts,
                                                 f'nuscenes_predictions_epoch_{self.current_epoch + 1}.json')
            
            elif self.config.get('eval_argoverse2', False):
                if self.is_distributed:
                    pred_dicts = self.model.module.pred_dicts
                else:
                    pred_dicts = self.model.pred_dicts
                
                if len(pred_dicts) > 0:
                    metric_results = self.model.compute_metrics_av2(
                        pred_dicts) if not self.is_distributed else self.model.module.compute_metrics_av2(pred_dicts)
                    aggregated_metrics.update({
                        'val_av2_official_minADE6': metric_results['min_ADE'],
                        'val_av2_official_minFDE6': metric_results['min_FDE'],
                        'val_av2_official_brier_minADE': metric_results['brier_min_ADE'],
                        'val_av2_official_brier_minFDE': metric_results['brier_min_FDE'],
                        'val_av2_official_miss_rate': metric_results['miss_rate']
                    })
                    
                    # 保存预测结果
                    self.logger.save_predictions(pred_dicts, f'av2_predictions_epoch_{self.current_epoch + 1}.json')
        
        return avg_loss, aggregated_metrics
    
    def _move_batch_to_device(self, batch):
        """将batch数据移动到设备"""
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict):
                    self._move_batch_to_device(value)
    
    def fit(self, train_loader, val_loader, max_epochs):
        """训练主循环"""
        if self.local_rank == 0:
            self.logger.log_text(f"Starting training for {max_epochs} epochs")
            self.logger.log_text(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # 设置分布式采样器的epoch
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
            
            # 训练
            start_time = time.time()
            train_loss = self.train_one_epoch(train_loader)
            train_time = time.time() - start_time
            
            # 验证
            start_time = time.time()
            val_loss, val_metrics = self.validate_one_epoch(val_loader)
            val_time = time.time() - start_time
            
            # 调度器步进
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 主进程记录和保存
            if self.local_rank == 0:
                # 准备所有指标
                all_metrics = {
                    'val_loss': val_loss,
                    'train_epoch_loss': train_loss,
                    'train_time_seconds': train_time,
                    'val_time_seconds': val_time,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # 添加验证指标（添加val_前缀）
                for key, value in val_metrics.items():
                    all_metrics[f'val_{key}'] = value
                
                # 记录指标
                self.logger.log_metrics(all_metrics, epoch=epoch + 1)
                
                # 记录epoch总结
                train_metrics = {'loss': train_loss, 'time': train_time}
                val_metrics_with_loss = val_metrics.copy()
                val_metrics_with_loss['loss'] = val_loss
                val_metrics_with_loss['time'] = val_time
                
                self.logger.log_epoch_summary(
                    epoch + 1,
                    train_metrics,
                    val_metrics_with_loss,
                    self.optimizer.param_groups[0]['lr']
                )
                
                # 检查是否是最佳模型
                current_metric = val_metrics.get('brier_fde', val_loss)
                is_best = current_metric < self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    self.logger.log_text(f"New best metric achieved: {current_metric:.4f}")
                
                # 保存检查点
                self.save_checkpoint(epoch + 1, current_metric, is_best)
        
        if self.local_rank == 0:
            self.logger.log_text("Training completed!")
            
            # 记录最佳结果
            best_metrics = self.logger.get_best_metrics('val_brier_fde', 'min')
            if best_metrics:
                self.logger.log_text(
                    f"Best validation brier_fde: {best_metrics['val_brier_fde']:.4f} at epoch {best_metrics['epoch']}")


if __name__ == '__main__':
    train()
