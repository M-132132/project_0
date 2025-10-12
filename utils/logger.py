import os
import json
import csv
from datetime import datetime
from typing import Dict, Any


class Logger:
    def __init__(self, log_dir: str, exp_name: str):
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.log_path = os.path.join(log_dir, exp_name)
        
        # 创建日志目录
        os.makedirs(self.log_path, exist_ok=True)
        
        # 初始化日志文件
        self.metrics_file = os.path.join(self.log_path, 'metrics.csv')
        self.train_log_file = os.path.join(self.log_path, 'train.log')
        self.config_file = os.path.join(self.log_path, 'config.json')
        
        # 初始化metrics CSV文件
        self.metrics_fieldnames = set()
        self.first_log = True
        
        print(f"Logger initialized. Logs will be saved to: {self.log_path}")
    
    def log_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        # 将OmegaConf转换为普通字典
        if hasattr(config, '_content'):
            # 处理OmegaConf对象
            config_dict = self._omegaconf_to_dict(config)
        else:
            config_dict = dict(config)
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to: {self.config_file}")
    
    def _omegaconf_to_dict(self, cfg):
        """递归转换OmegaConf对象为普通字典"""
        if hasattr(cfg, '_content'):
            result = {}
            for key, value in cfg.items():
                if hasattr(value, '_content'):
                    pass
                    # result[key] = self._omegaconf_to_dict(value)
                else:
                       result[key] = value
            return result
        else:
            return cfg
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int = None, step: int = None):
        """记录指标到CSV文件"""
        # 添加epoch和step信息
        log_data = {}
        if epoch is not None:
            log_data['epoch'] = epoch
        if step is not None:
            log_data['step'] = step
        log_data['timestamp'] = datetime.now().isoformat()
        log_data.update(metrics)
        
        # 更新fieldnames
        self.metrics_fieldnames.update(log_data.keys())
        
        # 写入CSV文件
        file_exists = os.path.exists(self.metrics_file)
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(self.metrics_fieldnames))
            
            # 如果是第一次写入或文件不存在，写入表头
            if not file_exists or self.first_log:
                writer.writeheader()
                self.first_log = False
            
            writer.writerow(log_data)
    
    def log_text(self, message: str, level: str = "INFO"):
        """记录文本日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}\n"
        
        # 写入日志文件
        with open(self.train_log_file, 'a') as f:
            f.write(log_message)
        
        # 同时打印到控制台
        print(f"[{level}] {message}")
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float], lr: float):
        """记录epoch总结"""
        summary = f"""
                    Epoch {epoch} Summary:
                    {'=' * 50}
                    Learning Rate: {lr:.6f}
                    Train Metrics:"""
        
        for key, value in train_metrics.items():
            summary += f"\n  {key}: {value:.4f}"
        
        summary += "\nValidation Metrics:"
        for key, value in val_metrics.items():
            summary += f"\n  {key}: {value:.4f}"
        
        summary += f"\n{'=' * 50}"
        
        self.log_text(summary)
    
    def save_predictions(self, predictions: list, filename: str = None):
        """保存预测结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"
        
        pred_file = os.path.join(self.log_path, filename)
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        self.log_text(f"Predictions saved to: {pred_file}")
    
    def load_metrics(self):
        """加载已记录的指标"""
        if not os.path.exists(self.metrics_file):
            return []
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                for key, value in row.items():
                    if key in ['epoch', 'step']:
                        try:
                            row[key] = int(value)
                        except (ValueError, TypeError):
                            pass
                    else:
                        try:
                            row[key] = float(value)
                        except (ValueError, TypeError):
                            pass
                metrics.append(row)
        
        return metrics
    
    def get_best_metrics(self, metric_name: str = 'val_brier_fde', mode: str = 'min'):
        """获取最佳指标"""
        metrics = self.load_metrics()
        if not metrics:
            return None
        
        # 过滤包含指定指标的记录
        valid_metrics = [m for m in metrics if metric_name in m and m[metric_name] is not None]
        if not valid_metrics:
            return None
        
        if mode == 'min':
            best_metric = min(valid_metrics, key=lambda x: x[metric_name])
        else:
            best_metric = max(valid_metrics, key=lambda x: x[metric_name])
        
        return best_metric