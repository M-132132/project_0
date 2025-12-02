import torch
from typing import Dict, Any
from .base_metric import BaseMetric

class SparsenessMetric(BaseMetric):
    """
    计算归因稀疏性（Sparseness）指标，使用基尼系数（Gini Index）。
    基尼系数越高，表示归因分数越集中在少数几个特征上，即越稀疏。
    """
    def compute(self, agent_scores: torch.Tensor, **kwargs) -> float:
        """
        计算基尼系数。
        
        Args:
            agent_scores (torch.Tensor): 聚合后的归因分数 [Batch, Num_Agents]。
            **kwargs: 未使用的其他参数。

        Returns:
            float: 批次平均的基尼系数值。
        """
        gini_scores = self._calculate_gini(agent_scores)
        return gini_scores.mean().item()

    def _calculate_gini(self, scores: torch.Tensor) -> torch.Tensor:
        """
        计算基尼系数 (Gini Index)。
        Args:
            scores: [Batch, Num_Agents]
        Returns:
            gini: [Batch]
        """
        # 修复：传入的 scores 已经是 abs() 过的，不再重复取绝对值
        x = scores + 1e-10
        sorted_x, _ = torch.sort(x, dim=1)
        
        n = sorted_x.size(1)
        index = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
        
        numerator = torch.sum((2 * index - n - 1) * sorted_x, dim=1)
        denominator = n * torch.sum(sorted_x, dim=1)
        
        return numerator / denominator