"""
轨迹预测归因计算主脚本

使用方法:
    python exps_scripts/exp_trajattr/compute_traj_attr.py --config-name autobot_attr
    python exps_scripts/exp_trajattr/compute_traj_attr.py --config-name wayformer_attr model_name=wayformer
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from models import build_model
from utils_datasets_traj import build_dataset
from utils.utils_train_traj import set_seed

from utils.path_manager import path_manager

# 导入归因计算框架
from utils_attr.traj_attr.base.traj_attr_base import TrajAttrBase


@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()),
            config_name="traj_attr_base")
def main(cfg: DictConfig) -> None:
    # 启用配置修改
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg = OmegaConf.merge(cfg, cfg.attribution)
    
    # 创建实验对象
    experiment = TrajAttrExperiment(cfg)
    
    # 运行实验
    attribution_results, analysis_results = experiment.run_attribution_experiment()
    
    # 打印成功信息
    print(f"\n✓ 实验成功完成！")
    print(f"✓ 结果保存在: {experiment.paths['base']}")

    # 如果启用可视化，提示运行可视化脚本
    if cfg.visualization.enable:
        print(f"\n💡 运行可视化脚本:")
        print(f"python exps_scripts/exp_trajattr/visualize_traj_attr.py "
              f"result_path={experiment.paths['base']}")
        
        
class TrajAttrExperiment:
    """轨迹预测归因计算实验类"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config.debug else 'cpu')
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 设置保存路径
        self.setup_save_paths()
        
        # 初始化组件
        self.model = None
        self.val_loader = None
        self.attributor = None
        
    def setup_save_paths(self):
        """设置保存路径"""
        # 使用 path_manager 解析基础路径
        base_dir = path_manager.resolve_path(self.config.save_config.base_dir)
        model_name = self.config.model_name
        dataset_name = self.config.dataset_name
        
        # 创建时间戳目录（如果启用）
        if self.config.save_config.create_timestamp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = f"{base_dir}/{model_name}_{dataset_name}/{timestamp}"
        else:
            exp_dir = f"{base_dir}/{model_name}_{dataset_name}"
            
        # 创建子目录
        self.paths = {
            'base': Path(exp_dir),
            'attributions': Path(exp_dir) / 'attributions',
            'numpy': Path(exp_dir) / 'attributions' / 'numpy',
            'heatmaps': Path(exp_dir) / 'attributions' / 'heatmaps',
            'statistics': Path(exp_dir) / 'attributions' / 'statistics',
            'visualizations': Path(exp_dir) / 'visualizations',
            'trajectory_plots': Path(exp_dir) / 'visualizations' / 'trajectory_plots',
            'map_plots': Path(exp_dir) / 'visualizations' / 'map_plots',
            'importance_analysis': Path(exp_dir) / 'visualizations' / 'importance_analysis',
            'reports': Path(exp_dir) / 'reports',
            'configs': Path(exp_dir) / 'configs'
        }
        
        # 创建所有目录
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        print(f"实验结果将保存到: {self.paths['base']}")
        
    def load_model_and_data(self):
        """加载模型和数据"""
        print(f"加载 {self.config.model_name} 模型...")
        
        # 构建模型
        self.model = build_model(self.config).to(self.device)
        
        # 加载检查点
        if self.config.ckpt_path and Path(path_manager.resolve_path(self.config.ckpt_path)).exists():
            weight_path = str(path_manager.resolve_path(self.config.ckpt_path))
            print(f"从检查点加载: {weight_path}")
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print("模型权重加载成功")
        else:
            print(f"警告: 检查点路径不存在 {self.config.ckpt_path}，使用随机权重")
            
        # 构建数据集
        print("加载验证数据集...")
        val_dataset = build_dataset(self.config, val=True)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.method.get('eval_batch_size', 2),
            num_workers=self.config.load_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"数据集加载完成: {len(val_dataset)} 个样本, {len(self.val_loader)} 个批次")
        
    def create_attributor(self):
        """创建归因计算器"""
        model_name = self.config.model_name.lower()
        
        # 直接使用完整的DictConfig，并传入保存路径
        # DictConfig 已经包含了 dirichlet_config, guided_ig_config, captum_config 等
        self.attributor = TrajAttrBase(self.model, self.config, self.paths)
            
        print(f"创建了 {model_name} 归因计算器（使用统一适配器）")
        
        
    def run_attribution_experiment(self):
        """运行完整的归因实验"""
        print("="*60)
        print(f"开始轨迹预测归因实验: {self.config.exp_name}")
        print(f"模型: {self.config.model_name}")
        print(f"数据集: {self.config.dataset_name}")
        print(f"归因方法: {self.config.attribution.methods}")
        print("="*60)
        
        # 保存实验配置
        self.save_experiment_config()
        
        # 加载模型和数据
        self.load_model_and_data()
        
        # 创建归因计算器
        self.create_attributor()
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 运行归因计算
        attribution_results = self.compute_batch_attributions()
        
        # 分析结果
        if self.config.analysis.generate_summary_statistics:
            analysis_results = self.analyze_attribution_results(attribution_results)
        else:
            analysis_results = {}
            
        # 生成实验报告
        self.generate_experiment_report(attribution_results, analysis_results)
        
        print("="*60)
        print("归因实验完成！")
        print(f"结果保存在: {self.paths['base']}")
        print("="*60)
        
        return attribution_results, analysis_results
        
    def compute_batch_attributions(self):
        """批量计算归因"""
        print(f"开始计算归因，限制批次数: {self.config.attribution.batch_limit}")
        
        attribution_results = []
        batch_limit = self.config.attribution.batch_limit
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='归因中')):
            if batch_idx >= batch_limit:
                break
            
            # 移动数据到设备
            self._move_to_device(batch, self.device)
            # 计算归因
            # 启用梯度计算
            torch.set_grad_enabled(True)
            
            # 创建元数据
            metadata = {
                'batch_idx': batch_idx,
                'model_name': self.config.model_name,
                'batch_size': self._get_batch_size(batch),
                'timestamp': datetime.now().isoformat()
            }
            
            # 计算归因
            batch_attributions = self.attributor.compute_and_save_attribution(
                batch,
                methods=self.config.attribution.methods,
                metadata=metadata
            )
            
            # 计算额外分析（如果是模型特定归因器）
            analysis = {}
            if hasattr(self.attributor, 'compute_feature_importance'):
                for method, attrs in batch_attributions.items():
                    try:
                        importance = self.attributor.compute_feature_importance(attrs, batch)
                        analysis[f'{method}_importance'] = importance
                    except Exception as e:
                        print(f"计算 {method} 特征重要性时出错: {e}")
            
            # 保存批次结果
            batch_result = {
                'metadata': metadata,
                'attributions': batch_attributions,
                'analysis': analysis
            }
            
            attribution_results.append(batch_result)
        
        print(f"归因计算完成，成功处理 {len(attribution_results)} 个批次")
        return attribution_results
        
    def analyze_attribution_results(self, attribution_results):
        """分析归因结果"""
        print("分析归因结果...")
        
        if not attribution_results:
            return {}
            
        analysis = {
            'summary': {
                'total_batches': len(attribution_results),
                'methods': self.config.attribution.methods,
                'model_name': self.config.model_name,
                'total_samples': sum(r['metadata']['batch_size'] for r in attribution_results)
            },
            'method_statistics': {},
            'importance_statistics': {}
        }
        
        # 统计每种方法的成功率
        for method in self.config.attribution.methods:
            success_count = sum(1 for r in attribution_results if method in r['attributions'])
            analysis['method_statistics'][method] = {
                'success_rate': success_count / len(attribution_results),
                'success_count': success_count,
                'total_count': len(attribution_results)
            }
        
        # 统计重要性分析
        if attribution_results[0]['analysis']:
            importance_keys = list(attribution_results[0]['analysis'].keys())
            for key in importance_keys:
                analysis['importance_statistics'][key] = {
                    'available_batches': sum(1 for r in attribution_results if key in r['analysis'])
                }
        
        # 保存分析结果
        analysis_path = self.paths['reports'] / 'attribution_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_types(analysis), f, indent=2, ensure_ascii=False)
            
        return analysis
        
    def save_experiment_config(self):
        """保存实验配置"""
        config_path = self.paths['configs'] / 'experiment_config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config=self.config, f=f)
            
    def generate_experiment_report(self, attribution_results, analysis_results):
        """生成实验报告"""
        report = {
            'experiment_info': {
                'exp_name': self.config.exp_name,
                'model_name': self.config.model_name,
                'dataset_name': self.config.dataset_name,
                'start_time': datetime.now().isoformat(),
                'config_file': str(self.paths['configs'] / 'experiment_config.yaml')
            },
            'attribution_settings': {
                'methods': self.config.attribution.methods,
                'batch_limit': self.config.attribution.batch_limit,
                'distance_type': self.config.attribution.distance_type
            },
            'results_summary': analysis_results.get('summary', {}),
            'method_performance': analysis_results.get('method_statistics', {}),
            'paths': {k: str(v) for k, v in self.paths.items()}
        }
        
        # 保存报告
        report_path = self.paths['reports'] / 'experiment_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 生成简要报告
        self._print_summary_report(report)
        
    def _print_summary_report(self, report):
        """打印简要报告"""
        print("\n" + "="*50)
        print("实验总结报告")
        print("="*50)
        print(f"实验名称: {report['experiment_info']['exp_name']}")
        print(f"模型: {report['experiment_info']['model_name']}")
        print(f"数据集: {report['experiment_info']['dataset_name']}")
        
        if 'results_summary' in report and report['results_summary']:
            summary = report['results_summary']
            print(f"处理批次: {summary.get('total_batches', 0)}")
            print(f"总样本数: {summary.get('total_samples', 0)}")
            
        print("\n归因方法性能:")
        for method, stats in report.get('method_performance', {}).items():
            print(f"  {method}: {stats['success_count']}/{stats['total_count']} "
                  f"({stats['success_rate']:.2%} 成功率)")
                  
        print(f"\n结果保存路径: {report['paths']['base']}")
        print("="*50)
        
    def _move_to_device(self, batch, device):
        """递归地将batch数据移动到指定设备"""
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif isinstance(value, dict):
                    self._move_to_device(value, device)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    batch[key] = [v.to(device) for v in value]
                    
    def _get_batch_size(self, batch):
        """获取批次大小"""
        if 'input_dict' in batch:
            for key, value in batch['input_dict'].items():
                if isinstance(value, torch.Tensor):
                    return value.size(0)
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        
        return 1
        
    def _convert_numpy_types(self, obj):
        """转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj


if __name__ == "__main__":
    main()
    