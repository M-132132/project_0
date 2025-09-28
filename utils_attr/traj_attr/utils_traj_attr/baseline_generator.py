"""
基线生成器

为归因计算生成各种类型的基线输入
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union


class BaselineGenerator:
    """
    基线生成器类
    
    提供多种基线生成策略，用于归因计算
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化基线生成器
        
        Args:
            device: 计算设备
        """
        self.device = device or torch.device('cpu')
    
    def zero_baseline(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        生成零基线
        
        Args:
            tensor: 输入张量
            
        Returns:
            baseline: 零基线张量
        """
        return torch.zeros_like(tensor)
    
    def random_baseline(self, tensor: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """
        生成随机噪声基线
        
        Args:
            tensor: 输入张量
            std: 噪声标准差
            
        Returns:
            baseline: 随机基线张量
        """
        return torch.randn_like(tensor) * std
    
    def mean_baseline(self, tensor: torch.Tensor, dim: Optional[Union[int, tuple]] = None) -> torch.Tensor:
        """
        生成均值基线
        
        Args:
            tensor: 输入张量
            dim: 计算均值的维度
            
        Returns:
            baseline: 均值基线张量
        """
        if dim is None:
            # 对除最后一个维度外的所有维度求均值
            dim = tuple(range(tensor.dim() - 1))
        
        mean_val = tensor.mean(dim=dim, keepdim=True)
        return mean_val.expand_as(tensor)
    
    def gaussian_baseline(self, tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """
        生成高斯分布基线
        
        Args:
            tensor: 输入张量
            mean: 高斯分布均值
            std: 高斯分布标准差
            
        Returns:
            baseline: 高斯基线张量
        """
        return torch.normal(mean, std, size=tensor.shape, device=tensor.device)
    
    def uniform_baseline(self, tensor: torch.Tensor, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
        """
        生成均匀分布基线
        
        Args:
            tensor: 输入张量
            low: 均匀分布下界
            high: 均匀分布上界
            
        Returns:
            baseline: 均匀分布基线张量
        """
        return torch.empty_like(tensor).uniform_(low, high)
    
    def masked_baseline(self, tensor: torch.Tensor, mask: torch.Tensor, 
                       fill_value: float = 0.0) -> torch.Tensor:
        """
        生成带掩码的基线
        
        Args:
            tensor: 输入张量
            mask: 掩码张量 (1表示保留原值，0表示填充)
            fill_value: 填充值
            
        Returns:
            baseline: 掩码基线张量
        """
        baseline = tensor.clone()
        baseline[mask == 0] = fill_value
        return baseline
    
    def trajectory_baseline(self, trajectory: torch.Tensor, baseline_type: str = 'zero') -> torch.Tensor:
        """
        为轨迹数据生成专用基线
        
        Args:
            trajectory: 轨迹张量 [B, N, T, F] 或 [B, T, F]
            baseline_type: 基线类型 ('zero', 'straight', 'stationary', 'random')
            
        Returns:
            baseline: 轨迹基线张量
        """
        if baseline_type == 'zero':
            return self.zero_baseline(trajectory)
        
        elif baseline_type == 'straight':
            # 直线轨迹基线（从起点到终点的直线）
            baseline = trajectory.clone()
            if baseline.dim() == 4:  # [B, N, T, F]
                start_pos = trajectory[:, :, 0:1, :2]  # 起点位置
                end_pos = trajectory[:, :, -1:, :2]   # 终点位置
                
                # 生成直线轨迹
                t_steps = torch.linspace(0, 1, trajectory.size(2), device=trajectory.device)
                t_steps = t_steps.view(1, 1, -1, 1)
                
                straight_traj = start_pos + t_steps * (end_pos - start_pos)
                baseline[:, :, :, :2] = straight_traj
                
                # 速度设为常值
                velocity = (end_pos - start_pos) / trajectory.size(2)
                baseline[:, :, :, 2:4] = velocity.expand(-1, -1, trajectory.size(2), -1)
                
                # 其他特征置零
                if trajectory.size(-1) > 4:
                    baseline[:, :, :, 4:] = 0
                    
            elif baseline.dim() == 3:  # [B, T, F]
                start_pos = trajectory[:, 0:1, :2]
                end_pos = trajectory[:, -1:, :2]
                
                t_steps = torch.linspace(0, 1, trajectory.size(1), device=trajectory.device)
                t_steps = t_steps.view(1, -1, 1)
                
                straight_traj = start_pos + t_steps * (end_pos - start_pos)
                baseline[:, :, :2] = straight_traj
                
                velocity = (end_pos - start_pos) / trajectory.size(1)
                baseline[:, :, 2:4] = velocity.expand(-1, trajectory.size(1), -1)
                
                if trajectory.size(-1) > 4:
                    baseline[:, :, 4:] = 0
            
            return baseline
        
        elif baseline_type == 'stationary':
            # 静止轨迹基线（保持在起点位置）
            baseline = trajectory.clone()
            
            if baseline.dim() == 4:  # [B, N, T, F]
                start_pos = trajectory[:, :, 0:1, :2]
                baseline[:, :, :, :2] = start_pos.expand(-1, -1, trajectory.size(2), -1)
                baseline[:, :, :, 2:] = 0  # 速度和其他特征置零
                
            elif baseline.dim() == 3:  # [B, T, F]
                start_pos = trajectory[:, 0:1, :2]
                baseline[:, :, :2] = start_pos.expand(-1, trajectory.size(1), -1)
                baseline[:, :, 2:] = 0
            
            return baseline
        
        elif baseline_type == 'random':
            return self.random_baseline(trajectory, std=0.1)
        
        else:
            return self.zero_baseline(trajectory)
    
    def map_baseline(self, map_data: torch.Tensor, baseline_type: str = 'zero') -> torch.Tensor:
        """
        为地图数据生成专用基线
        
        Args:
            map_data: 地图张量
            baseline_type: 基线类型 ('zero', 'random', 'mean')
            
        Returns:
            baseline: 地图基线张量
        """
        if baseline_type == 'zero':
            return self.zero_baseline(map_data)
        elif baseline_type == 'random':
            return self.random_baseline(map_data, std=0.05)
        elif baseline_type == 'mean':
            return self.mean_baseline(map_data)
        else:
            return self.zero_baseline(map_data)
    
    def generate_baseline(self, tensor: torch.Tensor, baseline_type: str = 'zero',
                         tensor_type: str = 'general', **kwargs) -> torch.Tensor:
        """
        通用基线生成函数
        
        Args:
            tensor: 输入张量
            baseline_type: 基线类型
            tensor_type: 张量类型 ('general', 'trajectory', 'map')
            **kwargs: 额外参数
            
        Returns:
            baseline: 基线张量
        """
        if tensor_type == 'trajectory':
            return self.trajectory_baseline(tensor, baseline_type)
        elif tensor_type == 'map':
            return self.map_baseline(tensor, baseline_type)
        else:
            # 通用张量处理
            if baseline_type == 'zero':
                return self.zero_baseline(tensor)
            elif baseline_type == 'random':
                return self.random_baseline(tensor, kwargs.get('std', 0.1))
            elif baseline_type == 'mean':
                return self.mean_baseline(tensor, kwargs.get('dim', None))
            elif baseline_type == 'gaussian':
                return self.gaussian_baseline(tensor, kwargs.get('mean', 0.0), kwargs.get('std', 1.0))
            elif baseline_type == 'uniform':
                return self.uniform_baseline(tensor, kwargs.get('low', -1.0), kwargs.get('high', 1.0))
            else:
                return self.zero_baseline(tensor)
    
    def generate_multiple_baselines(self, tensor: torch.Tensor, baseline_types: List[str],
                                  tensor_type: str = 'general', **kwargs) -> Dict[str, torch.Tensor]:
        """
        生成多种类型的基线
        
        Args:
            tensor: 输入张量
            baseline_types: 基线类型列表
            tensor_type: 张量类型
            **kwargs: 额外参数
            
        Returns:
            baselines: 基线字典
        """
        baselines = {}
        for baseline_type in baseline_types:
            baselines[baseline_type] = self.generate_baseline(
                tensor, baseline_type, tensor_type, **kwargs
            )
        return baselines