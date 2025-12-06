import torch
import copy
from typing import Dict, List, Optional
from .base_metric import BaseMetric

class LeRFMetric(BaseMetric):
    """
    计算 LeRF (Least Relevant First) 指标 (整车级)。
    (保留原有的整车 LeRF 代码，不做破坏性修改)
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
        
        # 获取按重要性升序排列的智能体索引（排除自车）
        # descending=False: 分数越低越靠前
        sorted_indices = self._get_sorted_indices_excluding_ego(
            agent_scores, ego_indices, descending=False
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
                # 如果是升序(LeRF)，把自车设为 +inf，排在最后
                # 如果是降序(MoRF)，把自车设为 -inf，排在最后
                b_scores[ego_indices[b]] = -float('inf') if descending else float('inf')
            indices = torch.argsort(b_scores, descending=descending)
            if ego_indices is not None:
                # 排除自车索引，使其不参与循环
                indices = indices[indices != ego_indices[b]]
            final_indices.append(indices)
        return final_indices

class LeRFFeatureMetric(LeRFMetric):
    """
    [新增] 特征级 LeRF 指标 (排除自车版)。
    逻辑：
    1. 排序：按归因分数【升序】排列 (最重要的排最后)。
    2. 排除自车：自车特征设为 +inf，确保它们排在列表最末尾，最后才被冻结。
    3. 操作：逐步冻结/扰动特征，观察预测变化。
    4. 预期：曲线应保持平缓 (Flat)，说明移除不重要特征对预测无影响。
    """
    def compute(self, 
                 batch: Dict, 
                 agent_scores: torch.Tensor, # shape [B, N, F]
                 target_key: str, 
                 steps: int, 
                 base_loss: float, 
                 ego_indices: Optional[torch.Tensor] = None,
                 noise_std: float = 1.0,     
                 metric_type: str = 'shift', # 'loss' or 'shift'
                 perturb_mode: str = 'freeze', # 默认为 freeze
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

        # 1. 展平分数: [B, N, F] -> [B, N*F]
        flat_scores = agent_scores.reshape(batch_size, -1)
        
        # === [核心逻辑] 排除自车 ===
        if ego_indices is not None:
            # LeRF 是升序排列，我们希望自车排在最后（最重要/最后被移除）
            # 所以将自车特征分数设为 正无穷 (+inf)
            for b in range(batch_size):
                ego_idx = ego_indices[b].item()
                start_feat = ego_idx * num_feats
                end_feat = start_feat + num_feats
                
                flat_scores[b, start_feat:end_feat] = float('inf')

        # 2. 排序 (descending=False，升序，分数低的在前)
        sorted_indices = torch.argsort(flat_scores, descending=False, dim=1)
        
        # 计算有效特征数 (排除自车)
        valid_features = (num_agents - 1) * num_feats if ego_indices is not None else num_agents * num_feats
        chunk_size = max(1, valid_features // steps)
        
        current_input = copy.deepcopy(batch['input_dict'])
        
        # 3. 循环冻结/扰动
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
            
            # 4. 计算指标
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