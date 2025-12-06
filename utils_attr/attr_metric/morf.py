import torch
import copy
from typing import Dict, List, Optional
from .base_metric import BaseMetric

class MoRFMetric(BaseMetric):
    """
    计算 MoRF (Most Relevant First) 指标。
    (保留原有的整车 MoRF 代码，不做修改)
    """
    def compute(self, 
                 batch: Dict, 
                 agent_scores: torch.Tensor, 
                 target_key: str, 
                 steps: int, 
                 base_loss: float, 
                 ego_indices: Optional[torch.Tensor] = None,
                 **kwargs) -> List[float]:
        
        batch_size, num_agents = agent_scores.shape
        curve = [base_loss]
        
        # 获取按重要性降序排列的智能体索引（排除自车）
        sorted_indices = self._get_sorted_indices_excluding_ego(
            agent_scores, ego_indices, descending=True
        )
        
        valid_agents_count = num_agents - (1 if ego_indices is not None else 0)
        chunk_size = max(1, valid_agents_count // steps)
        
        current_input = copy.deepcopy(batch['input_dict'])
        
        for i in range(0, valid_agents_count, chunk_size):
            end_idx = min(i + chunk_size, valid_agents_count)
            
            for b in range(batch_size):
                indices_to_mask = sorted_indices[b][i:end_idx]
                self._apply_mask(current_input, target_key, indices_to_mask, b)
            
            temp_batch = copy.deepcopy(batch)
            temp_batch['input_dict'] = current_input

            loss = self._compute_loss(temp_batch)
            curve.append(loss)
            
            if end_idx >= valid_agents_count:
                break
                
        return curve

    def _get_sorted_indices_excluding_ego(self, scores: torch.Tensor, ego_indices: Optional[torch.Tensor], descending: bool) -> List[torch.Tensor]:
        batch_size = scores.shape[0]
        final_indices = []
        for b in range(batch_size):
            b_scores = scores[b].clone()
            if ego_indices is not None:
                b_scores[ego_indices[b]] = -float('inf') if descending else float('inf')
            indices = torch.argsort(b_scores, descending=descending)
            if ego_indices is not None:
                indices = indices[indices != ego_indices[b]]
            final_indices.append(indices)
        return final_indices

class MoRFFeatureMetric(MoRFMetric):
    """
    [更新] 特征级 MoRF 指标 (排除自车版)。
    """
    def compute(self, 
                 batch: Dict, 
                 agent_scores: torch.Tensor, 
                 target_key: str, 
                 steps: int, 
                 base_loss: float, 
                 ego_indices: Optional[torch.Tensor] = None, # 必须传入自车索引
                 noise_std: float = 1.0,     
                 metric_type: str = 'shift', 
                 perturb_mode: str = 'freeze',
                 **kwargs) -> List[float]:
        
        batch_size, num_agents, num_feats = agent_scores.shape
        
        # 0. 准备基准
        if metric_type == 'shift':
            base_output = self._get_prediction(batch)
            probs = base_output['predicted_probability']
            trajs = base_output['predicted_trajectory'][..., :2]
            best_idxs = torch.argmax(probs, dim=1)
            base_traj = torch.stack([trajs[b, best_idxs[b]] for b in range(batch_size)])
            curve = [0.0] 
        else:
            base_traj = None
            curve = [base_loss]

        # 1. 展平分数
        flat_scores = agent_scores.reshape(batch_size, -1) # [B, N*F]
        
        # === [核心修复] 排除自车 ===
        if ego_indices is not None:
            # ego_indices: [B], 存储了每个样本中自车在 agents 列表里的 index
            # 我们需要把自车对应的 F 个特征的分数设为 -inf，让它们排在最后
            for b in range(batch_size):
                ego_idx = ego_indices[b].item()
                # 自车特征在 flat 数组中的范围
                start_feat = ego_idx * num_feats
                end_feat = start_feat + num_feats
                
                # 屏蔽自车
                flat_scores[b, start_feat:end_feat] = -float('inf')

        # 2. 排序 (现在自车肯定排在最后了)
        sorted_indices = torch.argsort(flat_scores, descending=True, dim=1)
        
        # 计算总特征数时，最好也减去自车的特征数，避免最后几步全是自车
        valid_features = (num_agents - 1) * num_feats if ego_indices is not None else num_agents * num_feats
        chunk_size = max(1, valid_features // steps)
        
        current_input = copy.deepcopy(batch['input_dict'])
        
        # 3. 循环冻结 (只循环 valid_features 范围)
        for i in range(0, valid_features, chunk_size):
            end_idx = min(i + chunk_size, valid_features)
            
            for b in range(batch_size):
                indices_to_mask = sorted_indices[b][i:end_idx]
                
                self._apply_feature_perturbation(
                    current_input, target_key, 
                    indices_to_mask, b, 
                    num_features=num_feats,
                    noise_std=noise_std, 
                    perturb_mode=perturb_mode 
                )
            
            # 4. 计算 Shift
            temp_batch = copy.deepcopy(batch)
            temp_batch['input_dict'] = current_input

            if metric_type == 'shift':
                val = self._compute_prediction_shift(temp_batch, base_traj)
            else:
                val = self._compute_loss(temp_batch)
                
            curve.append(val)
            
            if end_idx >= valid_features:
                break
                

                
        return curve