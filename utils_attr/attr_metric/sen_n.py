import torch
import numpy as np
import copy
from typing import Dict, Optional
from scipy.stats import pearsonr
from .base_metric import BaseMetric

class SenNMetric(BaseMetric):
    """
    计算 Sen-n (Sensitivity-n) 指标 (整车级)。
    (保留原代码不变)
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
            
            temp_batch = copy.deepcopy(batch)
            temp_batch['input_dict'] = current_input

            new_loss = self._compute_loss(temp_batch)
            loss_diffs.append(new_loss - base_loss)
            
        if len(attr_sums) > 1 and np.std(attr_sums) > 1e-6 and np.std(loss_diffs) > 1e-6:
            correlation, _ = pearsonr(attr_sums, loss_diffs)
        else:
            correlation = 0.0
            
        return {'pearson_corr': correlation, 'samples_count': len(attr_sums)}

class SenNFeatureMetric(SenNMetric):
    """
    [新增] 特征级 Sen-n 指标。
    支持 'freeze' 模式：随机冻结特征子集，验证分数与预测变化的相关性。
    """
    def compute(self, 
                 batch: Dict, 
                 agent_scores: torch.Tensor, # shape [B, N, F]
                 target_key: str, 
                 num_subsets: int, 
                 base_loss: float, 
                 ego_indices: Optional[torch.Tensor] = None,
                 noise_std: float = 1.0,
                 metric_type: str = 'shift',
                 perturb_mode: str = 'freeze', # [新增] 默认为冻结
                 **kwargs) -> Dict:
        
        batch_size, num_agents, num_feats = agent_scores.shape
        device = agent_scores.device
        
        # 1. 准备基准
        if metric_type == 'shift':
            base_output = self._get_prediction(batch)
            probs = base_output['predicted_probability']
            trajs = base_output['predicted_trajectory'][..., :2]
            best_idxs = torch.argmax(probs, dim=1)
            base_traj = torch.stack([trajs[b, best_idxs[b]] for b in range(batch_size)])
            base_val = 0.0
        else:
            base_traj = None
            base_val = base_loss

        attr_sums = []
        metric_diffs = []
        
        # 将分数展平 [B, N*F]
        flat_scores = agent_scores.reshape(batch_size, -1)
        
        # 获取所有有效特征的索引（排除自车）
        # 这一步是为了防止随机采样采到了自车特征
        valid_indices_map = [] # List[Tensor]
        for b in range(batch_size):
            all_indices = torch.arange(num_agents * num_feats, device=device)
            if ego_indices is not None:
                ego_idx = ego_indices[b].item()
                start = ego_idx * num_feats
                end = start + num_feats
                # 排除 [start, end) 范围的索引
                mask = torch.ones_like(all_indices, dtype=torch.bool)
                mask[start:end] = False
                valid_indices = all_indices[mask]
            else:
                valid_indices = all_indices
            valid_indices_map.append(valid_indices)

        # 2. 随机采样循环
        for _ in range(num_subsets):
            # 随机决定采样比例
            pct = np.random.uniform(0.01, 0.5)
            
            current_input = copy.deepcopy(batch['input_dict'])
            current_attr_sum = 0.0
            
            for b in range(batch_size):
                valid_indices = valid_indices_map[b]
                num_valid = len(valid_indices)
                k = max(1, int(num_valid * pct))
                
                # 从有效特征中随机选 k 个
                perm = torch.randperm(num_valid, device=device)
                selected_indices = valid_indices[perm[:k]]
                
                # A. 累加分数
                removed_scores = flat_scores[b, selected_indices]
                current_attr_sum += removed_scores.sum().item()
                
                # B. 执行扰动 (Freeze / Noise / Constant)
                # 这里将 perturb_mode 传递下去
                self._apply_feature_perturbation(
                    current_input, target_key, 
                    selected_indices, b, 
                    num_features=num_feats,
                    noise_std=noise_std,
                    perturb_mode=perturb_mode  # [关键]
                )
            
            attr_sums.append(current_attr_sum / batch_size)
            
            # C. 计算变化量
            temp_batch = copy.deepcopy(batch)
            temp_batch['input_dict'] = current_input

            if metric_type == 'shift':
                new_val = self._compute_prediction_shift(temp_batch, base_traj)
            else:
                new_val = self._compute_loss(temp_batch)
            
            # 记录变化幅度 (如果是 Shift，base_val=0，直接就是 Shift 值)
            metric_diffs.append(new_val - base_val)
            
        # 3. 计算相关性
        if len(attr_sums) > 1 and np.std(attr_sums) > 1e-9 and np.std(metric_diffs) > 1e-9:
            correlation, _ = pearsonr(attr_sums, metric_diffs)
        else:
            correlation = 0.0
            
        return {'pearson_corr': correlation, 'samples_count': len(attr_sums)}