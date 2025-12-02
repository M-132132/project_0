import torch
import copy
import numpy as np
from typing import Dict, Any

class BaseMetric:
    """
    归因评估指标的基类。
    定义了所有具体指标类必须遵循的通用接口。
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def compute(self, batch: Dict, attributions: Dict[str, torch.Tensor], **kwargs) -> Any:
        """
        计算指标的核心方法，必须由子类实现。
        
        Args:
            batch (Dict): 包含模型输入和真值的批次数据。
            attributions (Dict[str, torch.Tensor]): 归因分数张量字典。
            **kwargs: 其他特定于指标的参数。

        Returns:
            Any: 指标的计算结果。
        """
        raise NotImplementedError("每个指标子类必须实现 compute 方法。")

    def _compute_loss(self, batch: Dict) -> float:
        """
        一个辅助函数，用于计算给定批次数据的模型损失。
        """
        with torch.no_grad():
            # 创建一个副本以避免修改原始批次
            temp_batch = copy.deepcopy(batch)
            
            # 调用模型前向传播
            outputs, loss = self.model(temp_batch)
            
            loss_val = loss.mean().item()
            
            if np.isnan(loss_val) or np.isinf(loss_val):
                return 1e9  # 返回一个较大的数表示无效损失
            return loss_val

    def _apply_mask(self, input_dict: Dict, key: str, indices: torch.Tensor, batch_idx: int):
        """辅助函数，在输入的指定键上应用掩码，通常用于移除特征。"""
        mask_key = f"{key}_mask"
        if mask_key in input_dict:
            input_dict[mask_key][batch_idx, indices] = 0
        elif key in input_dict:
            # 如果没有专门的掩码键，直接将特征置零
            input_dict[key][batch_idx, indices] = 0