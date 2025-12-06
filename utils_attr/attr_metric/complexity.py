import torch
from typing import Dict, Any
from .base_metric import BaseMetric

class ComplexityMetric(BaseMetric):
    """
    计算归因复杂度（Complexity）指标，使用信息熵（Entropy）。
    熵越低，表示归因分布越确定（即越不均匀）。
    """
    def compute(self, agent_scores: torch.Tensor, **kwargs) -> float:
        """
        计算信息熵。

        Args:
            agent_scores (torch.Tensor): 聚合后的归因分数 [Batch, Num_Agents]。
            **kwargs: 未使用的其他参数。

        Returns:
            float: 批次平均的信息熵值。
        """
        entropy_scores = self._calculate_entropy(agent_scores)
        return entropy_scores.mean().item()

    def _calculate_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """
        计算信息熵 (Entropy)。
        """
        # 修复：传入的 scores 已经是 abs() 过的，不再重复取绝对值
        x_sum = scores.sum(dim=1, keepdim=True) + 1e-10
        probs = scores / x_sum
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy
    

class ComplexityFeatureMetric(ComplexityMetric):
    """
    [新增] 特征级复杂度指标。
    将所有特征展平后计算信息熵。
    """
    def compute(self, agent_scores: torch.Tensor, **kwargs) -> float:
        # agent_scores: [Batch, Num_Agents, Num_Features]
        batch_size = agent_scores.shape[0]
        
        # 1. 展平为 [Batch, Num_Agents * Num_Features]
        flat_scores = agent_scores.reshape(batch_size, -1)
        
        # 2. 调用父类的计算逻辑
        return super().compute(flat_scores, **kwargs)