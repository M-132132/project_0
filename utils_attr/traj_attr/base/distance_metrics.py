"""
轨迹预测距离度量函数

提供各种轨迹距离计算方法，用于从预测轨迹和真实轨迹计算标量损失
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DistanceMetrics:
    """
    轨迹预测距离度量类
    包含常用的轨迹距离计算方法
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def euclidean_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        计算欧几里得距离
        
        Args:
            pred: 预测轨迹 [..., 2]
            gt: 真实轨迹 [..., 2]
            
        Returns:
            distance: 欧几里得距离
        """
        return torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1))
    
    @staticmethod
    def l1_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        计算L1距离（曼哈顿距离）
        
        Args:
            pred: 预测轨迹 [..., 2]  
            gt: 真实轨迹 [..., 2]
            
        Returns:
            distance: L1距离
        """
        return torch.sum(torch.abs(pred - gt), dim=-1)
    
    @staticmethod
    def l2_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        计算L2距离（平方欧几里得距离）
        
        Args:
            pred: 预测轨迹 [..., 2]
            gt: 真实轨迹 [..., 2]
            
        Returns:
            distance: L2距离
        """
        return torch.sum((pred - gt) ** 2, dim=-1)
    
    def ade_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算平均位移误差（Average Displacement Error）
        
        Args:
            pred_trajs: 预测轨迹 [B, T, 2] 或 [B, M, T, 2]
            gt_trajs: 真实轨迹 [B, T, 2]
            mask: 有效时间步掩码 [B, T]
            
        Returns:
            ade_loss: ADE损失标量
        """
        if pred_trajs.dim() == 4:  # [B, M, T, 2] - 多模态预测
            B, M, T, _ = pred_trajs.shape  # B: batch_size, M: num_modes, T: time_steps, _: coordinate_dim(2)
            gt_trajs = gt_trajs.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 2]
        
        # 计算逐点距离
        distances = self.euclidean_distance(pred_trajs, gt_trajs)  # [B, (M,) T]
        
        # 应用掩码
        if mask is not None:
            if pred_trajs.dim() == 4:
                mask = mask.unsqueeze(1).expand(-1, M, -1)
            distances = distances * mask
            valid_steps = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            ade = distances.sum(dim=-1) / valid_steps.squeeze(-1)
        else:
            ade = distances.mean(dim=-1)  # [B, (M,)]
        
        if pred_trajs.dim() == 4:
            ade = ade.mean()  # 对所有模式和批次求平均
        else:
            ade = ade.mean()  # 对批次求平均
            
        return ade
    
    def fde_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算最终位移误差（Final Displacement Error）
        
        Args:
            pred_trajs: 预测轨迹 [B, T, 2] 或 [B, M, T, 2]
            gt_trajs: 真实轨迹 [B, T, 2]
            mask: 有效时间步掩码 [B, T]
            
        Returns:
            fde_loss: FDE损失标量
        """
        if mask is not None:
            # 找到每个序列的最后一个有效时间步
            last_valid_idx = mask.sum(dim=-1) - 1  # [B,]
            B = pred_trajs.size(0)
            
            if pred_trajs.dim() == 4:  # [B, M, T, 2]
                M = pred_trajs.size(1)
                pred_final = pred_trajs[torch.arange(B).unsqueeze(1), 
                                      torch.arange(M).unsqueeze(0), 
                                      last_valid_idx.unsqueeze(1)]  # [B, M, 2]
                gt_final = gt_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
                gt_final = gt_final.unsqueeze(1).expand(-1, M, -1)  # [B, M, 2]
            else:
                pred_final = pred_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
                gt_final = gt_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
        else:
            # 使用最后一个时间步
            if pred_trajs.dim() == 4:  # [B, M, T, 2]
                pred_final = pred_trajs[:, :, -1]  # [B, M, 2]
                gt_final = gt_trajs[:, -1].unsqueeze(1).expand(-1, pred_trajs.size(1), -1)  # [B, M, 2]
            else:
                pred_final = pred_trajs[:, -1]  # [B, 2]
                gt_final = gt_trajs[:, -1]  # [B, 2]
        
        # 计算最终位移距离
        fde = self.euclidean_distance(pred_final, gt_final)
        
        return fde.mean()
    
    def min_ade_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算最小ADE损失（用于多模态预测）
        
        Args:
            pred_trajs: 预测轨迹 [B, M, T, 2]
            gt_trajs: 真实轨迹 [B, T, 2]  
            mask: 有效时间步掩码 [B, T]
            
        Returns:
            min_ade_loss: 最小ADE损失标量
        """
        if pred_trajs.dim() != 4:
            return self.ade_loss(pred_trajs, gt_trajs, mask)
        
        B, M, T, _ = pred_trajs.shape  # B: batch_size, M: num_modes, T: time_steps, _: coordinate_dim(2)
        gt_trajs = gt_trajs.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 2]
        
        # 计算每个模式的ADE
        distances = self.euclidean_distance(pred_trajs, gt_trajs)  # [B, M, T]
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, M, -1)  # [B, M, T]
            distances = distances * mask
            valid_steps = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, M, 1]
            mode_ade = distances.sum(dim=-1, keepdim=True) / valid_steps  # [B, M, 1]
        else:
            mode_ade = distances.mean(dim=-1, keepdim=True)  # [B, M, 1]
        
        # 选择最小的ADE模式
        min_ade = mode_ade.min(dim=1)[0].squeeze(-1)  # [B,]
        
        return min_ade.mean()
    
    def min_fde_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算最小FDE损失（用于多模态预测）
        
        Args:
            pred_trajs: 预测轨迹 [B, M, T, 2]
            gt_trajs: 真实轨迹 [B, T, 2]
            mask: 有效时间步掩码 [B, T]
            
        Returns:
            min_fde_loss: 最小FDE损失标量
        """
        if pred_trajs.dim() != 4:
            return self.fde_loss(pred_trajs, gt_trajs, mask)
        
        B, M, T, _ = pred_trajs.shape  # B: batch_size, M: num_modes, T: time_steps, _: coordinate_dim(2)
        
        # 获取最终位置
        if mask is not None:
            last_valid_idx = mask.sum(dim=-1) - 1  # [B,]
            pred_final = pred_trajs[torch.arange(B).unsqueeze(1),
                                  torch.arange(M).unsqueeze(0),
                                  last_valid_idx.unsqueeze(1)]  # [B, M, 2]
            gt_final = gt_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
        else:
            pred_final = pred_trajs[:, :, -1]  # [B, M, 2]
            gt_final = gt_trajs[:, -1]  # [B, 2]
        
        gt_final = gt_final.unsqueeze(1).expand(-1, M, -1)  # [B, M, 2]
        
        # 计算每个模式的FDE
        mode_fde = self.euclidean_distance(pred_final, gt_final)  # [B, M]
        
        # 选择最小的FDE模式
        min_fde = mode_fde.min(dim=1)[0]  # [B,]
        
        return min_fde.mean()
    
    def displacement_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor,
                         loss_type: str = 'l2') -> torch.Tensor:
        """
        计算位移损失
        
        Args:
            pred_trajs: 预测轨迹 
            gt_trajs: 真实轨迹
            loss_type: 损失类型 ('l1', 'l2', 'smooth_l1')
            
        Returns:
            displacement_loss: 位移损失标量
        """
        if loss_type == 'l1':
            loss = F.l1_loss(pred_trajs, gt_trajs)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred_trajs, gt_trajs)
        elif loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_trajs, gt_trajs)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
        
        return loss
    
    def get_distance_function(self, distance_type: str = 'min_ade'):
        """
        获取距离计算函数
        
        Args:
            distance_type: 距离类型
            
        Returns:
            distance_function: 距离计算函数
        """
        distance_functions = {
            'ade': self.ade_loss,
            'fde': self.fde_loss,
            'min_ade': self.min_ade_loss,
            'min_fde': self.min_fde_loss,
            'l1': lambda p, g, m=None: self.displacement_loss(p, g, 'l1'),
            'l2': lambda p, g, m=None: self.displacement_loss(p, g, 'l2'),
            'smooth_l1': lambda p, g, m=None: self.displacement_loss(p, g, 'smooth_l1')
        }
        
        if distance_type not in distance_functions:
            raise ValueError(f"不支持的距离类型: {distance_type}. "
                           f"支持的类型: {list(distance_functions.keys())}")
        
        return distance_functions[distance_type]
    
    def min_ade_loss_batch_preserved(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算最小ADE损失，保留batch维度（用于归因计算）
        
        Args:
            pred_trajs: 预测轨迹 [B, M, T, 2] 或 [B, T, 2]
            gt_trajs: 真实轨迹 [B, T, 2]  
            mask: 有效时间步掩码 [B, T]
            
        Returns:
            min_ade_loss: 最小ADE损失 [B] 或 sum(保留batch信息的标量)
        """
        if pred_trajs.dim() == 3:  # [B, T, 2] - 单模态预测
            distances = self.euclidean_distance(pred_trajs, gt_trajs)  # [B, T]
            
            if mask is not None:
                distances = distances * mask
                valid_steps = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
                batch_ade = distances.sum(dim=-1, keepdim=True) / valid_steps  # [B, 1]
            else:
                batch_ade = distances.mean(dim=-1, keepdim=True)  # [B, 1]
            
            return batch_ade.squeeze(-1)  # 保留batch信息
        
        elif pred_trajs.dim() == 4:  # [B, M, T, 2] - 多模态预测
            B, M, T, _ = pred_trajs.shape  # B: batch_size, M: num_modes, T: time_steps, _: coordinate_dim(2)
            gt_trajs_expanded = gt_trajs.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 2]
            
            # 计算每个模式的ADE
            distances = self.euclidean_distance(pred_trajs, gt_trajs_expanded)  # [B, M, T]
            
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).expand(-1, M, -1)  # [B, M, T]
                distances = distances * mask_expanded
                valid_steps = mask_expanded.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, M, 1]
                mode_ade = distances.sum(dim=-1, keepdim=True) / valid_steps  # [B, M, 1]
            else:
                mode_ade = distances.mean(dim=-1, keepdim=True)  # [B, M, 1]
            
            # 选择最小的ADE模式，但保留batch维度
            min_ade, _ = mode_ade.squeeze(-1).min(dim=1)  # [B]
            return min_ade  # 保留batch信息
        
        else:
            raise ValueError(f"不支持的预测轨迹维度: {pred_trajs.dim()}")
    
    def min_fde_loss_batch_preserved(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算最小FDE损失，保留batch维度（用于归因计算）
        
        Args:
            pred_trajs: 预测轨迹 [B, M, T, 2] 或 [B, T, 2]
            gt_trajs: 真实轨迹 [B, T, 2]
            mask: 有效时间步掩码 [B, T]
            
        Returns:
            min_fde_loss: 最小FDE损失 sum(保留batch信息的标量)
        """
        if pred_trajs.dim() == 3:  # [B, T, 2] - 单模态预测
            # 获取最终位置
            if mask is not None:
                last_valid_idx = mask.sum(dim=-1) - 1  # [B,]
                B = pred_trajs.size(0)
                pred_final = pred_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
                gt_final = gt_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
            else:
                pred_final = pred_trajs[:, -1]  # [B, 2]
                gt_final = gt_trajs[:, -1]  # [B, 2]
            
            # 计算最终位移距离
            fde = self.euclidean_distance(pred_final, gt_final)  # [B]
            return fde  # 保留batch信息
        
        elif pred_trajs.dim() == 4:  # [B, M, T, 2] - 多模态预测
            B, M, T, _ = pred_trajs.shape  # B: batch_size, M: num_modes, T: time_steps, _: coordinate_dim(2)
            
            # 获取最终位置
            if mask is not None:
                last_valid_idx = mask.sum(dim=-1) - 1  # [B,]
                pred_final = pred_trajs[torch.arange(B).unsqueeze(1),
                                      torch.arange(M).unsqueeze(0),
                                      last_valid_idx.unsqueeze(1)]  # [B, M, 2]
                gt_final = gt_trajs[torch.arange(B), last_valid_idx]  # [B, 2]
            else:
                pred_final = pred_trajs[:, :, -1]  # [B, M, 2]
                gt_final = gt_trajs[:, -1]  # [B, 2]
            
            gt_final = gt_final.unsqueeze(1).expand(-1, M, -1)  # [B, M, 2]
            
            # 计算每个模式的FDE
            mode_fde = self.euclidean_distance(pred_final, gt_final)  # [B, M]
            
            # 选择最小的FDE模式，但保留batch维度
            min_fde, _ = mode_fde.min(dim=1)  # [B]
            return min_fde  # 保留batch信息
        
        else:
            raise ValueError(f"不支持的预测轨迹维度: {pred_trajs.dim()}")
    
    def get_distance_function_batch_preserved(self, distance_type: str = 'min_ade'):
        """
        获取保留batch维度的距离计算函数（用于归因计算）
        
        Args:
            distance_type: 距离类型
            
        Returns:
            distance_function: 保留batch维度的距离计算函数
        """
        distance_functions = {
            'min_ade': self.min_ade_loss_batch_preserved,
            'min_fde': self.min_fde_loss_batch_preserved,
        }
        
        if distance_type not in distance_functions:
            # 对于其他类型，使用普通函数但将结果乘以batch_size来保留batch信息
            regular_func = self.get_distance_function(distance_type)
            def batch_preserved_wrapper(pred_trajs, gt_trajs, mask=None):
                batch_size = gt_trajs.size(0)
                loss = regular_func(pred_trajs, gt_trajs, mask)
                return loss * batch_size  # 乘以batch_size来保留batch信息
            return batch_preserved_wrapper
        
        return distance_functions[distance_type]