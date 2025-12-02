import torch
import copy
from typing import Dict, List, Optional
from .base_metric import BaseMetric

class MoRFMetric(BaseMetric):
    """
    计算 MoRF (Most Relevant First) 指标。
    该指标通过逐步移除最相关的特征并观察模型性能的下降来评估归因的正确性。
    一个好的归因方法应该在移除最重要特征后导致性能急剧下降。
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
            
            # 修复：不再手动构建 temp_batch，而是复制原始 batch 并更新 input_dict
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