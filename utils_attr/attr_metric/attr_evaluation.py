import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy.stats import pearsonr
import copy
import inspect

class AttributionEvaluator:
    """
    轨迹预测归因评估器 (修复版 + 扩展版)
    
    包含指标：
    1. MoRF (Most Relevant First): 优先移除最重要特征，评估模型性能下降速度。
    2. LeRF (Least Relevant First): 优先移除最不重要特征，评估模型性能保持能力。
    3. Sen-n (Sensitivity-n): 评估归因分数与模型输出变化及其相关性。
    4. Sparseness (稀疏性): 使用基尼系数 (Gini Index) 衡量，值越高表示归因越聚焦。
    5. Complexity (复杂度): 使用信息熵 (Entropy) 衡量，值越低表示归因越确定。
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def evaluate(self, 
                 batch: Dict,                  # 输入数据批次
                 attributions: Dict[str, np.ndarray],   # 特征归因字典
                 metrics: List[str] = ['morf', 'lerf', 'sen_n'],  # 要计算的指标列表
                 target_key: str = 'obj_trajs',  # 目标键名，用于从归因字典中提取相关数据
                 steps: int = 10,              # 计算动态指标时的步数
                 num_subsets: int = 20) -> Dict[str, Any]:  # 子集数量，用于计算Sen-n指标
        
        results = {}
        if target_key not in attributions:
            return results

        # 准备数据
        device = next(self.model.parameters()).device
        attr_val = attributions[target_key]
        if isinstance(attr_val, np.ndarray):
            attr_tensor = torch.from_numpy(attr_val).to(device)
        else:
            attr_tensor = attr_val.to(device)
            
        # 聚合分数 [B, N, T, F] -> [B, N]
        # 对时间(T)和特征(F)维度求和，取绝对值表示重要性强度
        if attr_tensor.ndim >= 3:
            reduce_dims = tuple(range(2, attr_tensor.ndim))
            agent_scores = attr_tensor.abs().sum(dim=reduce_dims)
        else:
            agent_scores = attr_tensor.abs()

        # --- 计算无需模型推理的静态指标 ---
        
        # 4. 计算 Sparseness (Gini Index)
        if 'sparseness' in metrics:
            # 计算每个样本的 Gini，然后取 Batch 平均
            gini_scores = self._calculate_gini(agent_scores)
            results['sparseness'] = gini_scores.mean().item()

        # 5. 计算 Complexity (Entropy)
        if 'complexity' in metrics:
            # 计算每个样本的 Entropy，然后取 Batch 平均
            entropy_scores = self._calculate_entropy(agent_scores)
            results['complexity'] = entropy_scores.mean().item()

        # --- 计算需要模型推理的动态指标 (MoRF, LeRF, Sen-n) ---
        
        # 如果没有动态指标需求，直接返回，节省时间
        dynamic_metrics = {'morf', 'lerf', 'sen_n'}
        if not dynamic_metrics.intersection(set(metrics)):
            return results

        # 1. 获取基础 Loss
        # 使用 copy 防止污染原始 batch
        temp_batch = copy.copy(batch) 
        base_loss = self._compute_loss(temp_batch)
        results['base_loss'] = base_loss

        # 2. 获取自车索引 (Ego Index)，用于保护
        ego_indices = batch['input_dict'].get('track_index_to_predict', None)

        if 'morf' in metrics:
            results['morf'] = self._compute_morf_lerf(
                batch, agent_scores, target_key, steps, mode='morf', 
                base_loss=base_loss, ego_indices=ego_indices
            )

        if 'lerf' in metrics:
            results['lerf'] = self._compute_morf_lerf(
                batch, agent_scores, target_key, steps, mode='lerf', 
                base_loss=base_loss, ego_indices=ego_indices
            )

        if 'sen_n' in metrics:
            results['sen_n'] = self._compute_sen_n(
                batch, agent_scores, target_key, num_subsets, 
                base_loss=base_loss, ego_indices=ego_indices
            )

        return results

    def _calculate_gini(self, scores: torch.Tensor) -> torch.Tensor:
        """
        计算基尼系数 (Gini Index)
        Args:
            scores: [Batch, Num_Agents]
        Returns:
            gini: [Batch]
        """
        # 1. 预处理：加上微小值避免全0，确保非负
        x = scores + 1e-10
        
        # 2. 排序 (Ascending)
        # sorted_x: [Batch, N]
        sorted_x, _ = torch.sort(x, dim=1)
        
        # 3. 计算 Gini
        # 公式: G = [ sum_{i=1}^n (2i - n - 1) x_i ] / [ n * sum(x_i) ]
        n = sorted_x.size(1)
        # index: [1, 2, ..., n]
        index = torch.arange(1, n + 1, device=x.device).float()
        
        numerator = torch.sum((2 * index - n - 1) * sorted_x, dim=1)
        denominator = n * torch.sum(sorted_x, dim=1)
        
        return numerator / denominator

    def _calculate_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """
        计算信息熵 (Entropy)
        Args:
            scores: [Batch, Num_Agents]
        Returns:
            entropy: [Batch]
        """
        # 1. 归一化为概率分布 (Sum = 1)
        # p_i = x_i / sum(x)
        x_sum = scores.sum(dim=1, keepdim=True) + 1e-10
        probs = scores / x_sum
        
        # 2. 计算熵: H(x) = - sum(p * log(p))
        # 避免 log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        return entropy

    def _compute_morf_lerf(self, batch, scores, key, steps, mode, base_loss, ego_indices=None):
        batch_size, num_agents = scores.shape
        curve = [base_loss]
        
        sorted_indices = self._get_sorted_indices_excluding_ego(
            scores, ego_indices, descending=(mode == 'morf')
        )
        
        valid_agents_count = num_agents - 1 if ego_indices is not None else num_agents
        chunk_size = max(1, valid_agents_count // steps)
        
        current_input = copy.deepcopy(batch['input_dict'])
        
        for i in range(0, valid_agents_count, chunk_size):
            end_idx = min(i + chunk_size, valid_agents_count)
            
            for b in range(batch_size):
                indices_to_mask = sorted_indices[b][i:end_idx]
                self._apply_mask(current_input, key, indices_to_mask, b)
            
            temp_batch = copy.copy(batch)
            temp_batch['input_dict'] = current_input
            
            loss = self._compute_loss(temp_batch)
            curve.append(loss)
            
            if end_idx >= valid_agents_count:
                break
                
        return curve

    def _get_sorted_indices_excluding_ego(self, scores, ego_indices, descending=True):
        batch_size, num_agents = scores.shape
        final_indices = []
        
        for b in range(batch_size):
            b_scores = scores[b].clone()
            
            if ego_indices is not None:
                ego_idx = ego_indices[b]
                if descending:
                    b_scores[ego_idx] = -float('inf')
                else:
                    b_scores[ego_idx] = float('inf')
            
            indices = torch.argsort(b_scores, descending=descending)
            
            if ego_indices is not None:
                indices = indices[:-1]
                
            final_indices.append(indices)
            
        return final_indices

    def _compute_sen_n(self, batch, scores, key, num_subsets, base_loss, ego_indices=None):
        batch_size, num_agents = scores.shape
        device = scores.device
        
        attr_sums = []
        loss_diffs = []
        
        valid_count = num_agents - 1 if ego_indices is not None else num_agents
        
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
                
                removed_scores = scores[b, indices_to_mask]
                current_attr_sum += removed_scores.sum().item()
                
                self._apply_mask(current_input, key, indices_to_mask, b)
            
            attr_sums.append(current_attr_sum / batch_size)
            
            temp_batch = copy.copy(batch)
            temp_batch['input_dict'] = current_input
            new_loss = self._compute_loss(temp_batch)
            
            loss_diffs.append(new_loss - base_loss)
            
        if len(attr_sums) > 1 and np.std(attr_sums) > 1e-6 and np.std(loss_diffs) > 1e-6:
            correlation, _ = pearsonr(attr_sums, loss_diffs)
        else:
            correlation = 0.0
            
        return {'pearson_corr': correlation, 'samples_count': len(attr_sums)}

    def _compute_loss(self, batch):
        with torch.no_grad():
            try:
                outputs = self.model(batch)
            except TypeError:
                import inspect
                forward_params = inspect.signature(self.model.forward).parameters
                kwargs = {}
                if 'static_dict' in forward_params: kwargs['static_dict'] = batch.get('static_dict')
                if 'target_trajs' in forward_params: kwargs['target_trajs'] = batch.get('target_trajs')
                outputs = self.model(batch['input_dict'], **kwargs)

            loss_val = 0.0
            
            if isinstance(outputs, tuple):
                if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
                    loss_val = outputs[1].item()
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss_val = outputs['loss'].mean().item()
            elif isinstance(outputs, torch.Tensor):
                loss_val = outputs.mean().item()
            
            if np.isnan(loss_val) or np.isinf(loss_val):
                return 1e9
            return loss_val

    def _apply_mask(self, input_dict, key, indices, batch_idx):
        mask_key = f"{key}_mask"
        if mask_key in input_dict:
            input_dict[mask_key][batch_idx, indices] = 0
        else:
            if key in input_dict:
                input_dict[key][batch_idx, indices] = 0