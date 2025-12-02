import torch
import numpy as np
import copy
from typing import Dict, Optional
from scipy.stats import pearsonr
from .base_metric import BaseMetric

class SenNMetric(BaseMetric):
    """
    计算 Sen-n (Sensitivity-n) 指标。
    该指标通过随机移除n个特征子集，评估归因分数总和与模型性能变化之间的相关性。
    一个好的归因方法应该表现出强正相关性。
    """
    def compute(self, 
                 batch: Dict, 
                 agent_scores: torch.Tensor, 
                 target_key: str, 
                 num_subsets: int, 
                 base_loss: float, 
                 ego_indices: Optional[torch.Tensor] = None,
                 **kwargs) -> Dict:
        
        batch_size, num_agents = agent_scores.shape
        device = agent_scores.device
        
        attr_sums = []
        loss_diffs = []
        
        valid_count = num_agents - (1 if ego_indices is not None else 0)
        
        for _ in range(num_subsets):
            pct = np.random.uniform(0.01, 0.5)
            k = max(1, int(valid_count * pct))
            
            current_input = copy.deepcopy(batch['input_dict'])
            current_attr_sum = 0.0
            
            for b in range(batch_size):
                candidates = torch.arange(num_agents, device=device)
                if ego_indices is not None:
                    ego_idx = ego_indices[b]
                    candidates = candidates[candidates != ego_idx]
                
                perm = torch.randperm(len(candidates), device=device)
                indices_to_mask = candidates[perm[:k]]
                
                removed_scores = agent_scores[b, indices_to_mask]
                current_attr_sum += removed_scores.sum().item()
                
                self._apply_mask(current_input, target_key, indices_to_mask, b)
            
            attr_sums.append(current_attr_sum / batch_size)
            
            # 修复：不再手动构建 temp_batch，而是复制原始 batch 并更新 input_dict
            temp_batch = copy.deepcopy(batch)
            temp_batch['input_dict'] = current_input

            new_loss = self._compute_loss(temp_batch)
            
            loss_diffs.append(new_loss - base_loss)
            
        if len(attr_sums) > 1 and np.std(attr_sums) > 1e-6 and np.std(loss_diffs) > 1e-6:
            correlation, _ = pearsonr(attr_sums, loss_diffs)
        else:
            correlation = 0.0
            
        return {'pearson_corr': correlation, 'samples_count': len(attr_sums)}