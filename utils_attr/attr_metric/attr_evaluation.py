import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import copy

from .morf import MoRFMetric
from .lerf import LeRFMetric
from .sen_n import SenNMetric
from .sparseness import SparsenessMetric
from .complexity import ComplexityMetric

class AttributionEvaluator:
    """
    轨迹预测归因评估器。
    
    该类作为归因指标计算的调度中心。它本身不实现具体的指标计算逻辑，
    而是动态地实例化并调用相应的指标类（如 MoRFMetric, SparsenessMetric 等）来完成计算。
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self._metric_calculators = {
            'morf': MoRFMetric(model),
            'lerf': LeRFMetric(model),
            'sen_n': SenNMetric(model),
            'sparseness': SparsenessMetric(model),
            'complexity': ComplexityMetric(model),
        }

    def evaluate(self, 
                 batch: Dict,                  # 输入数据批次
                 attributions: Dict[str, np.ndarray],   # 特征归因字典
                 metrics: List[str] = ['morf', 'lerf', 'sen_n'],  # 要计算的指标列表
                 target_key: str = 'obj_trajs',  # 目标键名，用于从归因字典中提取相关数据

                 
                 steps: int = 10,              # 计算动态指标时的步数
                 num_subsets: int = 20) -> Dict[str, Any]:  # 子集数量，用于计算Sen-n指标
        
        results = {}
        if not metrics or target_key not in attributions:
            return results

        # 1. 准备通用数据
        device = next(self.model.parameters()).device
        attr_val = attributions[target_key]
        attr_tensor = torch.from_numpy(attr_val).to(device) if isinstance(attr_val, np.ndarray) else attr_val.to(device)
            
        # 2. 聚合归因分数: [B, N, T, F] -> [B, N]
        if attr_tensor.ndim >= 3:
            reduce_dims = tuple(range(2, attr_tensor.ndim))
            agent_scores = attr_tensor.abs().sum(dim=reduce_dims)
        else:
            agent_scores = attr_tensor.abs()

        # 3. 准备动态指标所需的参数
        dynamic_metrics = {'morf', 'lerf', 'sen_n'}
        needs_dynamic = any(metric in dynamic_metrics for metric in metrics)
        base_loss = None
        ego_indices = None

        if needs_dynamic:
            # 仅在需要时计算 base_loss
            base_loss = self._compute_loss(batch)
            results['base_loss'] = base_loss
            # 获取自车索引，用于在扰动时保护自车
            ego_indices = batch['input_dict'].get('track_index_to_predict', None)

        # 4. 遍历并计算所有请求的指标
        for metric_name in metrics:
            if metric_name in self._metric_calculators:
                calculator = self._metric_calculators[metric_name]
                
                # 准备传递给 compute 方法的参数
                params = {
                    'batch': batch,
                    'agent_scores': agent_scores,
                    'target_key': target_key,
                    'steps': steps,
                    'num_subsets': num_subsets,
                    'base_loss': base_loss,
                    'ego_indices': ego_indices,
                }
                
                try:
                    # 调用具体指标的 compute 方法
                    results[metric_name] = calculator.compute(**params)
                except Exception as e:
                    print(f"警告: 计算指标 '{metric_name}' 时失败: {e}")
                    results[metric_name] = None
            else:
                print(f"警告: 未知的归因指标 '{metric_name}'")

        return results

    def _compute_loss(self, batch: Dict) -> float:
        """辅助函数，用于计算给定批次数据的模型损失。"""
        with torch.no_grad():
            # 创建一个副本以避免修改原始批次
            temp_batch = copy.deepcopy(batch)
            
            # 调用模型前向传播
            outputs, loss = self.model(temp_batch)
            
            loss_val = loss.mean().item()
            
            if np.isnan(loss_val) or np.isinf(loss_val):
                return 1e9  # 返回一个较大的数表示无效损失
            return loss_val