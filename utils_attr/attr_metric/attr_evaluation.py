import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import copy

# 导入所有基础指标和特征级指标
from .morf import MoRFMetric, MoRFFeatureMetric
from .lerf import LeRFMetric, LeRFFeatureMetric
from .sen_n import SenNMetric, SenNFeatureMetric
from .sparseness import SparsenessMetric, SparsenessFeatureMetric # [新增导入]
from .complexity import ComplexityMetric, ComplexityFeatureMetric # [新增导入]

class AttributionEvaluator:
    """
    轨迹预测归因评估器。
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        # 注册所有计算器
        self._metric_calculators = {
            # 动态指标
            'morf': MoRFMetric(model),
            'morf_feature': MoRFFeatureMetric(model),
            
            'lerf': LeRFMetric(model),
            'lerf_feature': LeRFFeatureMetric(model),
            
            'sen_n': SenNMetric(model),
            'sen_n_feature': SenNFeatureMetric(model),
            
            # 静态分布指标
            'sparseness': SparsenessMetric(model),
            'sparseness_feature': SparsenessFeatureMetric(model), # [注册]
            
            'complexity': ComplexityMetric(model),
            'complexity_feature': ComplexityFeatureMetric(model), # [注册]
        }

    def evaluate(self, 
                 batch: Dict,                  
                 attributions: Dict[str, np.ndarray],   
                 metrics: List[str] = ['morf', 'lerf', 'sen_n'],
                 target_key: str = 'obj_trajs',
                 steps: int = 10,              
                 num_subsets: int = 20,        
                 
                 # === 控制参数 ===
                 evaluation_mode: str = 'agent',  # 'agent' 或 'feature'
                 noise_std: float = 1.0,
                 metric_type: str = 'loss',
                 perturb_mode: str = 'constant'
                 ) -> Dict[str, Any]:
        
        results = {}
        if not metrics or target_key not in attributions:
            return results

        # 1. 准备通用数据
        device = next(self.model.parameters()).device
        attr_val = attributions[target_key]
        attr_tensor = torch.from_numpy(attr_val).to(device) if isinstance(attr_val, np.ndarray) else attr_val.to(device)
            
        # 2. 聚合归因分数
        if evaluation_mode == 'feature':
            # 特征级评估: [B, N, T, F] -> [B, N, F]
            # (如果是 4D 张量，对时间维度 T 求和)
            if attr_tensor.ndim == 4:
                agent_scores = attr_tensor.abs().sum(dim=2) 
                # 仅取前两个特征 (x, y)，防止 mask 干扰
                if agent_scores.shape[-1] > 2:
                    agent_scores = agent_scores[..., :2]
            else:
                agent_scores = attr_tensor.abs()
        else:
            # 整车级评估: [B, N, T, F] -> [B, N]
            if attr_tensor.ndim >= 3:
                reduce_dims = tuple(range(2, attr_tensor.ndim))
                agent_scores = attr_tensor.abs().sum(dim=reduce_dims)
            else:
                agent_scores = attr_tensor.abs()

        # 3. 准备参数
        base_loss = None
        ego_indices = batch['input_dict'].get('track_index_to_predict', None)

        if metric_type == 'loss':
            base_loss = self._compute_loss(batch)
            results['base_loss'] = base_loss

        # 4. 遍历计算指标
        for metric_name in metrics:
            # 自动切换计算器名称
            calculator_name = metric_name
            if evaluation_mode == 'feature':
                # 简单的后缀映射逻辑
                feature_name = f"{metric_name}_feature"
                if feature_name in self._metric_calculators:
                    calculator_name = feature_name
            
            if calculator_name in self._metric_calculators:
                calculator = self._metric_calculators[calculator_name]
                
                # 构建参数包
                params = {
                    'batch': batch,
                    'agent_scores': agent_scores,
                    'target_key': target_key,
                    'steps': steps,
                    'num_subsets': num_subsets,
                    'base_loss': base_loss,
                    'ego_indices': ego_indices,
                    'noise_std': noise_std,
                    'metric_type': metric_type,
                    'perturb_mode': perturb_mode
                }
                
                try:
                    results[metric_name] = calculator.compute(**params)
                except Exception as e:
                    print(f"警告: 计算指标 '{metric_name}' (使用 {calculator_name}) 时失败: {e}")
                    import traceback
                    traceback.print_exc()
                    results[metric_name] = None
            else:
                print(f"警告: 未找到对应的计算器 '{calculator_name}' (原始请求: {metric_name})")

        return results

    def _compute_loss(self, batch: Dict) -> float:
        with torch.no_grad():
            temp_batch = copy.deepcopy(batch)
            outputs, loss = self.model(temp_batch)
            loss_val = loss.mean().item()
            if np.isnan(loss_val) or np.isinf(loss_val):
                return 1e9
            return loss_val