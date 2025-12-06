import torch
import copy
import numpy as np
from typing import Dict, Any

class BaseMetric:
    """归因评估指标基类"""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def _get_prediction(self, batch: Dict):
        """获取模型原始预测"""
        with torch.no_grad():
            temp_batch = copy.deepcopy(batch)
            output, _ = self.model(temp_batch)
            return output

    def _compute_prediction_shift(self, batch: Dict, base_traj: torch.Tensor) -> float:
        """计算预测轨迹位移 (ADE Shift)"""
        with torch.no_grad():
            output = self._get_prediction(batch)
            probs = output['predicted_probability']
            trajs = output['predicted_trajectory'][..., :2]
            best_idxs = torch.argmax(probs, dim=1)
            
            # 提取最佳模态轨迹
            current_traj = torch.stack([trajs[b, best_idxs[b]] for b in range(trajs.shape[0])])
            
            # 计算位移
            shift = torch.norm(current_traj - base_traj, dim=-1).mean().item()
            return shift

    def _compute_loss(self, batch: Dict) -> float:
        """计算 Loss"""
        with torch.no_grad():
            temp_batch = copy.deepcopy(batch)
            outputs, loss = self.model(temp_batch)
            loss_val = loss.mean().item()
            if np.isnan(loss_val) or np.isinf(loss_val): return 1e9
            return loss_val

    def _apply_mask(self, input_dict: Dict, key: str, indices: torch.Tensor, batch_idx: int):
        """传统整车 Mask (置零)"""
        mask_key = f"{key}_mask"
        if mask_key in input_dict:
            input_dict[mask_key][batch_idx, indices] = 0
        elif key in input_dict:
            input_dict[key][batch_idx, indices] = 0

    def _apply_feature_perturbation(self, input_dict: Dict, key: str, 
                                    indices: torch.Tensor, batch_idx: int, 
                                    num_features: int = 2,
                                    noise_std: float = 1.0,
                                    perturb_mode: str = 'constant'):
        """
        [核心修改] 特征级扰动/冻结
        perturb_mode:
          'freeze': [新] 轨迹冻结。让特征相对于初始时刻保持不变 (消除相对运动)。
          'constant': 平移扰动。
          'noise': 噪声扰动。
        """
        data = input_dict[key] # [B, N, T, F]
        device = data.device
        
        # 解析索引
        agent_indices = indices // num_features
        feature_indices = indices % num_features
        
        T = data.shape[2]
        
        if perturb_mode == 'freeze':
            # === 轨迹冻结逻辑 ===
            # 1. 获取 t=0 时刻的初始值 (相对坐标的起点)
            # shape: [Num_Selected]
            initial_values = data[batch_idx, agent_indices, 0, feature_indices]
            
            # 2. 将初始值广播到所有时间步 (消除随时间的变化)
            # shape: [Num_Selected, T]
            frozen_traj = initial_values.unsqueeze(1).repeat(1, T)
            
            # 3. 覆盖原数据
            data[batch_idx, agent_indices, :, feature_indices] = frozen_traj
            
        elif perturb_mode == 'constant':
            # 平移逻辑 (之前讨论的)
            offset = torch.randn(len(indices), 1, device=device) * noise_std
            data[batch_idx, agent_indices, :, feature_indices] += offset
            
        else: # 'noise'
            noise = torch.randn(len(indices), T, device=device) * noise_std
            data[batch_idx, agent_indices, :, feature_indices] += noise