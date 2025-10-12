"""
模型适配器系统

为不同轨迹预测模型提供统一的输入输出接口，支持归因计算
"""

import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from ..base.distance_metrics import DistanceMetrics


class BaseModelAdapter(ABC):
    """
    模型适配器基类
    """
    
    def __init__(self, model, model_name: str = None):
        """
        初始化模型适配器
        
        Args:
            model: 模型实例
            model_name: 模型名称
        """
        # 初始化模型相关属性
        self.model = model  # 传入的模型实例
        self.model_name = model_name or self.__class__.__name__  # 设置模型名称，如果未提供则使用类名
        self.device = next(model.parameters()).device  # 获取模型所在的设备(CPU/GPU)
        self.distance_metrics = DistanceMetrics()  # 初始化距离度量工具类
    
    @abstractmethod
    def get_attribution_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        从batch中提取需要计算归因的输入张量
        
        Args:
            batch: 原始批量数据
            
        Returns:
            attribution_inputs: 需要归因的输入字典
        """
        pass
    
    @abstractmethod
    def get_static_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        从batch中提取不需要梯度的静态输入
        
        Args:
            batch: 原始批量数据
            
        Returns:
            static_inputs: 静态输入字典
        """
        pass
    
    @abstractmethod
    def reconstruct_batch(self, attribution_inputs: Dict, static_inputs: Dict) -> Dict:
        """
        重构模型可以接受的batch格式
        
        Args:
            attribution_inputs: 归因输入字典
            static_inputs: 静态输入字典
            
        Returns:
            batch: 重构的batch
        """
        pass
    
    @abstractmethod
    def extract_prediction(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """
        从模型输出中提取预测结果
        
        Args:
            model_output: 模型原始输出
            
        Returns:
            prediction: 标准化的预测结果字典
        """
        pass
    
    def forward_with_loss(self, attribution_inputs: Dict, static_inputs: Dict,
                         target_trajs: torch.Tensor = None, distance_type: str = 'min_ade') -> torch.Tensor:
        """
        执行前向传播并返回用于归因的标量损失
        
        Args:
            attribution_inputs: 归因输入
            static_inputs: 静态输入
            target_trajs: 目标轨迹
            distance_type: 距离计算类型
            
        Returns:
            scalar_loss: 标量损失
        """
        # 重构batch
        batch = self.reconstruct_batch(attribution_inputs, static_inputs)
        
        # 前向传播
        prediction, loss = self.model(batch)
        
        # # 如果模型本身返回loss，直接使用
        # if loss is not None and isinstance(loss, torch.Tensor):
        #     return loss
        
        # 否则需要根据预测结果和目标计算损失
        pred_dict = self.extract_prediction([prediction, loss])
        
        if target_trajs is not None:
            # 使用外部提供的目标轨迹
            return self._compute_distance_loss(pred_dict, target_trajs, distance_type)
        else:
            # 尝试从静态输入中提取目标轨迹
            if 'center_gt_trajs' in static_inputs:
                gt_trajs = static_inputs['center_gt_trajs'][..., :2]  # 只取x,y坐标
                return self._compute_distance_loss(pred_dict, gt_trajs, distance_type)
            else:
                # 从attribution_inputs中选择一个tensor来创建连接的损失
                for tensor in attribution_inputs.values():
                    if isinstance(tensor, torch.Tensor):
                        return (tensor.sum() * 0.0) + 0.001
                return torch.tensor(0.001, requires_grad=True, device=self.device)
    
    def _compute_distance_loss(self, pred_dict: Dict, gt_trajs: torch.Tensor, 
                              distance_type: str = 'min_ade') -> torch.Tensor:
        """
        使用distance_metrics计算预测轨迹与真实轨迹的距离损失
        
        Args:
            pred_dict: 预测结果字典
            gt_trajs: 真实轨迹 [B, T, 2]
            distance_type: 距离计算类型
            
        Returns:
            distance_loss: 距离损失（保留batch信息的标量）
        """
        pred_trajs = pred_dict.get('predicted_trajectory', None)
        if pred_trajs is None:
            # 返回一个与gt_trajs连接的零损失，保持计算图和batch_size相关
            batch_size = gt_trajs.size(0)
            return (gt_trajs * 0.0).sum() + 0.001 * batch_size
        
        # 提取位置信息（只取x, y坐标）
        if pred_trajs.dim() == 4:  # [B, M, T, F]
            pred_pos = pred_trajs[..., :2]  # [B, M, T, 2]
        elif pred_trajs.dim() == 3:  # [B, T, F]
            pred_pos = pred_trajs[..., :2]  # [B, T, 2]
        else:
            # 异常情况处理
            batch_size = gt_trajs.size(0)
            return (pred_trajs.sum() * 0.0 + gt_trajs.sum() * 0.0) + 0.001 * batch_size
        
        # 使用distance_metrics计算距离损失（保留batch维度）
        distance_func = self.distance_metrics.get_distance_function_batch_preserved(distance_type)
        
        # 注意：这里可能需要掩码信息，暂时使用None
        mask = None  # 可以从static_inputs中获取掩码信息
        distance_loss = distance_func(pred_pos, gt_trajs, mask)
        return distance_loss


class AutoBotAdapter(BaseModelAdapter):
    """
    AutoBot模型适配器
    """
    
    def get_attribution_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """提取需要归因的输入"""
        inputs = {}
        input_dict = batch['input_dict']
        
        # 智能体轨迹数据
        if 'obj_trajs' in input_dict:
            inputs['obj_trajs'] = input_dict['obj_trajs'].detach().requires_grad_(True)
            
        # 地图数据  
        if 'map_polylines' in input_dict:
            inputs['map_polylines'] = input_dict['map_polylines'].detach().requires_grad_(True)
            
        return inputs
    
    def get_static_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """提取静态输入"""
        inputs = {}
        input_dict = batch['input_dict']
        
        # 掩码和索引信息
        for key in ['obj_trajs_mask', 'track_index_to_predict', 'map_polylines_mask']:
                inputs[key] = input_dict[key]
        
        # 真实轨迹信息
        for key in ['center_gt_trajs', 'center_gt_trajs_mask', 'center_gt_final_valid_idx']:
            if key in input_dict:
                inputs[key] = input_dict[key]
                
        return inputs
    
    def reconstruct_batch(self, attribution_inputs: Dict, static_inputs: Dict) -> Dict:
        """重构AutoBot的batch格式"""
        input_dict = {}
        input_dict.update(attribution_inputs)
        input_dict.update(static_inputs)
        
        return {'input_dict': input_dict}
    
    def extract_prediction(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """提取AutoBot的预测结果"""
        prediction, loss = model_output
        return {
            'predicted_trajectory': prediction.get('predicted_trajectory'),
            'predicted_probability': prediction.get('predicted_probability'),
            'loss': loss
        }


class WayformerAdapter(BaseModelAdapter):
    """
    Wayformer模型适配器
    """
    
    def get_attribution_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """提取需要归因的输入"""
        inputs = {}
        input_dict = batch['input_dict']
        
        # 智能体轨迹数据
        if 'obj_trajs' in input_dict:
            inputs['obj_trajs'] = input_dict['obj_trajs'].detach().requires_grad_(True)
            
        # 地图数据
        if 'map_polylines' in input_dict:
            inputs['map_polylines'] = input_dict['map_polylines'].detach().requires_grad_(True)
            
        return inputs
    
    def get_static_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """提取静态输入"""
        inputs = {}
        input_dict = batch['input_dict']
        
        # 掩码和索引信息
        for key in ['obj_trajs_mask', 'track_index_to_predict', 'map_polylines_mask']:
            if key in input_dict:
                inputs[key] = input_dict[key]
        
        # 真实轨迹信息
        for key in ['center_gt_trajs', 'center_gt_trajs_mask', 'center_gt_final_valid_idx']:
            if key in input_dict:
                inputs[key] = input_dict[key]
                
        return inputs
    
    def reconstruct_batch(self, attribution_inputs: Dict, static_inputs: Dict) -> Dict:
        """重构Wayformer的batch格式"""
        input_dict = {}
        input_dict.update(attribution_inputs)  
        input_dict.update(static_inputs)
        
        return {'input_dict': input_dict}
    
    def extract_prediction(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """提取Wayformer的预测结果"""
        prediction, loss = model_output
        return {
            'predicted_trajectory': prediction.get('predicted_trajectory'),
            'predicted_probability': prediction.get('predicted_probability'), 
            'loss': loss
        }


class MTRAdapter(BaseModelAdapter):
    """
    MTR模型适配器
    """
    
    def get_attribution_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """提取需要归因的输入"""
        inputs = {}
        
        # MTR可能直接使用batch，需要检查具体的输入字段
        for key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask']:
            if key in batch:
                tensor = batch[key]
                if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float64]:
                    inputs[key] = tensor.detach().requires_grad_(True)
        
        # 如果batch中有input_dict，也检查里面的内容
        if 'input_dict' in batch:
            input_dict = batch['input_dict']
            for key in ['obj_trajs', 'map_polylines']:
                if key in input_dict:
                    tensor = input_dict[key]
                    if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float64]:
                        inputs[key] = tensor.detach().requires_grad_(True)
        
        return inputs
    
    def get_static_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """提取静态输入"""
        inputs = {}
        
        # 复制所有非浮点型的输入
        for key, value in batch.items():
            if key != 'input_dict' and not (isinstance(value, torch.Tensor) and 
                                          value.dtype in [torch.float32, torch.float64]):
                inputs[key] = value
        
        # 处理input_dict中的静态输入
        if 'input_dict' in batch:
            input_dict = batch['input_dict']
            inputs['input_dict'] = {}
            for key, value in input_dict.items():
                if key not in ['obj_trajs', 'map_polylines']:  # 这些已经在attribution_inputs中了
                    inputs['input_dict'][key] = value
        
        return inputs
    
    def reconstruct_batch(self, attribution_inputs: Dict, static_inputs: Dict) -> Dict:
        """重构MTR的batch格式"""
        batch = {}
        batch.update(static_inputs)
        
        # 如果有input_dict，更新其中的归因输入
        if 'input_dict' in batch:
            batch['input_dict'].update(attribution_inputs)
        else:
            # 直接添加归因输入到顶级
            batch.update(attribution_inputs)
        
        return batch
    
    def extract_prediction(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """提取MTR的预测结果"""
        prediction, loss = model_output
        return {
            'predicted_trajectory': prediction.get('predicted_trajectory'),
            'predicted_probability': prediction.get('predicted_probability'),
            'loss': loss
        }


class ModelAdapterFactory:
    """
    模型适配器工厂类
    """
    
    _adapters = {
        'autobot': AutoBotAdapter,
        'autobotego': AutoBotAdapter,
        'wayformer': WayformerAdapter,
        'mtr': MTRAdapter,
        'smart': MTRAdapter,  # SMART可能使用类似MTR的接口
    }
    
    @classmethod
    def create_adapter(cls, model, model_name: str = None) -> BaseModelAdapter:
        """
        创建模型适配器
        
        Args:
            model: 模型实例
            model_name: 模型名称
            
        Returns:
            adapter: 模型适配器实例
        """
        # 自动检测模型类型
        if model_name is None:
            model_class_name = model.__class__.__name__.lower()

            # 尝试匹配已知的模型类型
            for known_type in cls._adapters.keys():
                if known_type in model_class_name:
                    model_name = known_type
                    break
        
        if model_name is None:
            # 默认使用通用适配器（基于AutoBot）
            model_name = 'autobot'
        
        model_name = model_name.lower()
        
        if model_name not in cls._adapters:
            raise ValueError(f"不支持的模型类型: {model_name}. "
                           f"支持的类型: {list(cls._adapters.keys())}")
        
        # 从适配器字典中获取指定模型名称对应的适配器类
        adapter_class = cls._adapters[model_name]
        return adapter_class(model, model_name)
    
    @classmethod
    def register_adapter(cls, model_name: str, adapter_class: type):
        """
        注册新的模型适配器
        
        Args:
            model_name: 模型名称
            adapter_class: 适配器类
        """
        cls._adapters[model_name.lower()] = adapter_class
        
    @classmethod  
    def get_supported_models(cls) -> List[str]:
        """获取支持的模型列表"""
        return list(cls._adapters.keys())