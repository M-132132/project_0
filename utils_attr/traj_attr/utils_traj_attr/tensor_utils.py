"""
张量处理工具

提供张量操作、转换和处理的实用函数
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional


class TensorUtils:
    """
    张量处理工具类
    """
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        将张量转换为numpy数组
        
        Args:
            tensor: 输入张量
            
        Returns:
            array: numpy数组
        """
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.cpu().numpy()
    
    @staticmethod
    def from_numpy(array: np.ndarray, device: torch.device = None, 
                   requires_grad: bool = False) -> torch.Tensor:
        """
        将numpy数组转换为张量
        
        Args:
            array: numpy数组
            device: 目标设备
            requires_grad: 是否需要梯度
            
        Returns:
            tensor: 张量
        """
        tensor = torch.from_numpy(array)
        if device is not None:
            tensor = tensor.to(device)
        if requires_grad:
            tensor = tensor.requires_grad_(True)
        return tensor
    
    @staticmethod
    def normalize_tensor(tensor: torch.Tensor, method: str = 'minmax',
                        dim: Optional[Union[int, tuple]] = None) -> torch.Tensor:
        """
        标准化张量
        
        Args:
            tensor: 输入张量
            method: 标准化方法 ('minmax', 'zscore', 'l2')
            dim: 标准化的维度
            
        Returns:
            normalized_tensor: 标准化后的张量
        """
        if method == 'minmax':
            if dim is None:
                min_val = tensor.min()
                max_val = tensor.max()
            else:
                min_val = tensor.min(dim=dim, keepdim=True)[0]
                max_val = tensor.max(dim=dim, keepdim=True)[0]
            
            # 避免除零
            range_val = max_val - min_val
            range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
            
            return (tensor - min_val) / range_val
        
        elif method == 'zscore':
            if dim is None:
                mean_val = tensor.mean()
                std_val = tensor.std()
            else:
                mean_val = tensor.mean(dim=dim, keepdim=True)
                std_val = tensor.std(dim=dim, keepdim=True)
            
            # 避免除零
            std_val = torch.where(std_val == 0, torch.ones_like(std_val), std_val)
            
            return (tensor - mean_val) / std_val
        
        elif method == 'l2':
            if dim is None:
                norm = tensor.norm()
            else:
                norm = tensor.norm(dim=dim, keepdim=True)
            
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            return tensor / norm
        
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    @staticmethod
    def apply_mask(tensor: torch.Tensor, mask: torch.Tensor, 
                  fill_value: float = 0.0) -> torch.Tensor:
        """
        应用掩码到张量
        
        Args:
            tensor: 输入张量
            mask: 掩码张量 (1表示保留，0表示掩盖)
            fill_value: 填充值
            
        Returns:
            masked_tensor: 掩码后的张量
        """
        return torch.where(mask.bool(), tensor, torch.full_like(tensor, fill_value))
    
    @staticmethod
    def interpolate_trajectory(trajectory: torch.Tensor, target_length: int,
                              method: str = 'linear') -> torch.Tensor:
        """
        插值轨迹到指定长度
        
        Args:
            trajectory: 输入轨迹 [..., T, F]
            target_length: 目标长度
            method: 插值方法 ('linear', 'nearest')
            
        Returns:
            interpolated_trajectory: 插值后的轨迹
        """
        original_shape = trajectory.shape
        T = original_shape[-2]
        F = original_shape[-1]
        
        if T == target_length:
            return trajectory
        
        # 重塑为 [batch_size, T, F]
        batch_dims = original_shape[:-2]
        batch_size = np.prod(batch_dims) if batch_dims else 1
        traj_reshaped = trajectory.view(batch_size, T, F)
        
        # 转置为 [batch_size, F, T] 用于插值
        traj_transposed = traj_reshaped.transpose(1, 2)
        
        # 插值
        if method == 'linear':
            interpolated = torch.nn.functional.interpolate(
                traj_transposed, size=target_length, mode='linear', align_corners=True
            )
        elif method == 'nearest':
            interpolated = torch.nn.functional.interpolate(
                traj_transposed, size=target_length, mode='nearest'
            )
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        # 转回原始格式并重塑
        interpolated = interpolated.transpose(1, 2)
        output_shape = batch_dims + (target_length, F)
        
        return interpolated.view(output_shape)
    
    @staticmethod
    def smooth_trajectory(trajectory: torch.Tensor, window_size: int = 3,
                         method: str = 'gaussian') -> torch.Tensor:
        """
        平滑轨迹
        
        Args:
            trajectory: 输入轨迹 [..., T, F]
            window_size: 平滑窗口大小
            method: 平滑方法 ('gaussian', 'average', 'median')
            
        Returns:
            smoothed_trajectory: 平滑后的轨迹
        """
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("窗口大小必须是大于等于3的奇数")
        
        original_shape = trajectory.shape
        T = original_shape[-2]
        F = original_shape[-1]
        
        # 重塑为 [batch_size, F, T]
        batch_dims = original_shape[:-2]
        batch_size = np.prod(batch_dims) if batch_dims else 1
        traj_reshaped = trajectory.view(batch_size, F, T)
        
        if method == 'gaussian':
            # 创建高斯核
            sigma = window_size / 6.0
            kernel_size = window_size
            kernel = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, -1).to(trajectory.device)
            
            # 应用卷积
            padding = kernel_size // 2
            smoothed = torch.nn.functional.conv1d(
                traj_reshaped, kernel.expand(F, -1, -1), 
                padding=padding, groups=F
            )
        
        elif method == 'average':
            # 平均滤波
            kernel = torch.ones(1, 1, window_size, device=trajectory.device) / window_size
            padding = window_size // 2
            smoothed = torch.nn.functional.conv1d(
                traj_reshaped, kernel.expand(F, -1, -1),
                padding=padding, groups=F
            )
        
        else:
            raise ValueError(f"不支持的平滑方法: {method}")
        
        # 重塑回原始形状
        return smoothed.view(original_shape)
    
    @staticmethod
    def compute_trajectory_features(trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算轨迹特征
        
        Args:
            trajectory: 轨迹张量 [..., T, 2] (位置信息)
            
        Returns:
            features: 特征字典
        """
        features = {}
        
        # 速度 (一阶差分)
        velocity = torch.diff(trajectory, dim=-2)  # [..., T-1, 2]
        features['velocity'] = velocity
        
        # 速率 (速度的模长)
        speed = torch.norm(velocity, dim=-1, keepdim=True)  # [..., T-1, 1]
        features['speed'] = speed
        
        # 加速度 (二阶差分)
        if velocity.size(-2) > 1:
            acceleration = torch.diff(velocity, dim=-2)  # [..., T-2, 2]
            features['acceleration'] = acceleration
            
            # 加速度大小
            accel_magnitude = torch.norm(acceleration, dim=-1, keepdim=True)
            features['acceleration_magnitude'] = accel_magnitude
        
        # 方向角 (相对于x轴的角度)
        direction = torch.atan2(velocity[..., 1:2], velocity[..., 0:1])  # [..., T-1, 1]
        features['direction'] = direction
        
        # 转向角 (方向角的变化)
        if direction.size(-2) > 1:
            turning_angle = torch.diff(direction, dim=-2)
            # 处理角度跳跃
            turning_angle = torch.atan2(torch.sin(turning_angle), torch.cos(turning_angle))
            features['turning_angle'] = turning_angle
        
        # 累积距离
        distances = torch.norm(velocity, dim=-1)  # [..., T-1]
        cumulative_distance = torch.cumsum(distances, dim=-1)  # [..., T-1]
        features['cumulative_distance'] = cumulative_distance.unsqueeze(-1)
        
        return features
    
    @staticmethod
    def batch_tensor_dict(tensor_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        将张量字典列表批量化
        
        Args:
            tensor_dicts: 张量字典列表
            
        Returns:
            batched_dict: 批量化的张量字典
        """
        if not tensor_dicts:
            return {}
        
        batched_dict = {}
        for key in tensor_dicts[0].keys():
            tensors = [d[key] for d in tensor_dicts if key in d]
            if tensors:
                batched_dict[key] = torch.stack(tensors, dim=0)
        
        return batched_dict
    
    @staticmethod
    def split_tensor_dict(tensor_dict: Dict[str, torch.Tensor], 
                         batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        将批量化的张量字典拆分
        
        Args:
            tensor_dict: 批量化的张量字典
            batch_size: 批量大小
            
        Returns:
            tensor_dicts: 张量字典列表
        """
        tensor_dicts = []
        
        for i in range(batch_size):
            single_dict = {}
            for key, tensor in tensor_dict.items():
                single_dict[key] = tensor[i]
            tensor_dicts.append(single_dict)
        
        return tensor_dicts


# 从 utils_attr_cal.py 融合的函数

def l1_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    计算两个张量的L1距离
    
    Args:
        t1: 第一个张量
        t2: 第二个张量
        
    Returns:
        distance: L1距离
    """
    return torch.sum(torch.abs(t1 - t2))


def calculate_straight_line_path(x_input: torch.Tensor, x_baseline: torch.Tensor, 
                                steps: int) -> List[torch.Tensor]:
    """
    计算从基线到输入的直线路径
    
    Args:
        x_input: 输入张量
        x_baseline: 基线张量
        steps: 插值步数
        
    Returns:
        path: 路径点列表
    """
    alphas = torch.linspace(0.0, 1.0, steps=steps + 1, device=x_input.device)
    res = [x_baseline + alpha * (x_input - x_baseline) for alpha in alphas]
    return res


def compute_decision_net_gradients(forward_fn: callable, inputs: torch.Tensor, 
                                  target: torch.Tensor = None, 
                                  before_sigmoid: bool = False) -> torch.Tensor:
    """
    计算决策网络的梯度
    
    Args:
        forward_fn: 前向传播函数
        inputs: 输入张量
        target: 目标张量（可选）
        before_sigmoid: 是否使用sigmoid之前的值
        
    Returns:
        grads: 梯度张量
    """
    with torch.autograd.set_grad_enabled(True):
        inputs = inputs.requires_grad_(True)
        output = forward_fn(inputs, before_sigmoid)
        
        num_examples = output.shape[0]
        
        if target is not None:
            target = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
            target = (target,) if isinstance(target, int) else target
            tar_output = output[(slice(None), *target)]
        else:
            tar_output = output.sum()
        
        grads = torch.autograd.grad(tar_output, inputs)[0]
    
    return grads


def proc_ori_attr_np(attrs_ori: np.ndarray) -> np.ndarray:
    """
    处理原始归因值，保留绝对值更大的属性
    
    Args:
        attrs_ori: 原始归因数组
        
    Returns:
        attrs_ori_scaled: 处理后的归因数组
    """
    attrs_ori_abs = np.abs(attrs_ori)
    max_indices = np.argmax(attrs_ori_abs, axis=-1)
    attrs_ori_sum = attrs_ori[np.arange(attrs_ori.shape[0])[:, np.newaxis],
                              np.arange(attrs_ori.shape[1]), max_indices]
    
    # 使用简化的标准化（避免依赖 utils_save）
    attrs_min = attrs_ori_sum.min()
    attrs_max = attrs_ori_sum.max()
    if attrs_max != attrs_min:
        attrs_ori_scaled = (attrs_ori_sum - attrs_min) / (attrs_max - attrs_min)
    else:
        attrs_ori_scaled = attrs_ori_sum
    
    return attrs_ori_scaled


def quan_attr(attrs: np.ndarray, attr_quantile: float) -> np.ndarray:
    """
    根据分位数过滤归因值
    
    Args:
        attrs: 归因数组
        attr_quantile: 分位数阈值
        
    Returns:
        filtered_attrs: 过滤后的归因数组
    """
    threshold = np.quantile(attrs, attr_quantile)
    return attrs * (attrs > threshold)


def get_tuple_blob_info(metadata: dict) -> tuple:
    """
    从元数据中提取blob信息
    
    Args:
        metadata: 元数据字典
        
    Returns:
        blob_info: blob信息元组
    """
    if hasattr(metadata, 'blob_info') or 'blob_info' in metadata:
        b_dict = metadata['blob_info']
    else:
        b_dict = metadata
    
    blob_info = (
        b_dict['xs'], 
        b_dict['ys'], 
        b_dict['sizes'], 
        b_dict['covs'],
        b_dict['features'], 
        b_dict['spatial_style']
    )
    return blob_info