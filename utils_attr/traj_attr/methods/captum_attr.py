"""
Captum归因方法集成

将Captum库中的归因方法适配到轨迹预测模型中
"""

import torch
from typing import Dict, Tuple, Any, Optional
import captum.attr as capattr
from ..utils_traj_attr.baseline_generator import BaselineGenerator


class CaptumAttribution:
    """
    Captum归因方法集成类
    
    支持Captum库中的多种归因方法
    """
    
    def __init__(self, attr_base, method_name: str, distance_type: str = 'min_ade', **kwargs):
        """
        初始化Captum归因计算器
        
        Args:
            attr_base: TrajAttrBase实例
            method_name: Captum方法名称
            distance_type: 距离计算类型
            **kwargs: 方法特定参数
        """
        self.attr_base = attr_base
        self.model = attr_base.model
        self.device = attr_base.device
        self.method_name = method_name
        self.distance_type = distance_type
        self.kwargs = kwargs
        
        # 获取距离计算函数
        self.distance_fn = attr_base.distance_metrics.get_distance_function(distance_type)
        
        # 初始化基线生成器
        self.baseline_generator = BaselineGenerator(device=self.device)
        
        # 支持的Captum方法映射
        self.captum_methods = {
            'IntegratedGradients': capattr.IntegratedGradients,
            'DeepLift': capattr.DeepLift,
            'DeepLiftShap': capattr.DeepLiftShap,
            'GradientShap': capattr.GradientShap,
            'InputXGradient': capattr.InputXGradient,
            'Saliency': capattr.Saliency,
            'GuidedBackprop': capattr.GuidedBackprop,
            'GuidedGradCam': capattr.GuidedGradCam,
            'LRP': capattr.LRP,
            'ShapleyValueSampling': capattr.ShapleyValueSampling,
            'FeaturePermutation': capattr.FeaturePermutation,
            'Occlusion': capattr.Occlusion,
            'KernelShap': capattr.KernelShap,
            'Lime': capattr.Lime
        }
        
        if method_name not in self.captum_methods:
            raise ValueError(f"不支持的Captum方法: {method_name}. "
                           f"支持的方法: {list(self.captum_methods.keys())}")
        
        # 创建前向传播函数包装器
        self.forward_func = self._create_forward_wrapper()
        
        # 创建归因方法实例
        self.attr_method = self._create_attribution_method()
    
    def _create_forward_wrapper(self):
        """
        创建用于Captum的前向传播包装器
        
        Returns:
            forward_func: 包装后的前向传播函数
        """
        def forward_func(*input_tensors):
            """
            Captum格式的前向传播函数
            
            Args:
                *input_tensors: 输入张量序列
                
            Returns:
                distance_loss: 标量距离损失
            """
            # 重构归因输入字典
            attribution_inputs = {}
            
            # 从缓存的输入键列表中重构
            if hasattr(self, '_input_keys_cache'):
                for i, key in enumerate(self._input_keys_cache):
                    if i < len(input_tensors):
                        attribution_inputs[key] = input_tensors[i]
            
            # 检查batch_size是否匹配，如果不匹配则扩展static_inputs
            static_inputs = self._static_inputs_cache
            attr_batch_size = None
            static_batch_size = None
            
            if attribution_inputs and static_inputs:
                # 获取attribution_inputs的当前batch_size
                for key, tensor in attribution_inputs.items():
                    if isinstance(tensor, torch.Tensor):
                        attr_batch_size = tensor.size(0)
                        break
                
                # 获取static_inputs的原始batch_size
                for key, tensor in static_inputs.items():
                    if isinstance(tensor, torch.Tensor):
                        static_batch_size = tensor.size(0)
                        break
                
                # 如果batch_size不匹配（通常是IntegratedGradients等方法导致的）
                if (attr_batch_size is not None and static_batch_size is not None and 
                    attr_batch_size != static_batch_size):
                    
                    # 计算重复倍数
                    repeat_factor = attr_batch_size // static_batch_size
                    if repeat_factor > 1 and attr_batch_size % static_batch_size == 0:
                        # 扩展static_inputs以匹配attribution_inputs的batch_size
                        expanded_static_inputs = {}
                        for key, value in static_inputs.items():
                            if isinstance(value, torch.Tensor):
                                # 沿batch维度重复
                                expanded_value = value.repeat(repeat_factor, *([1] * (value.dim() - 1)))
                                expanded_static_inputs[key] = expanded_value
                            else:
                                expanded_static_inputs[key] = value
                        static_inputs = expanded_static_inputs
            
            # 处理GT数据的扩展
            gt_cache = getattr(self, '_gt_cache', None)
            if (gt_cache is not None and attr_batch_size is not None and static_batch_size is not None and 
                attr_batch_size != static_batch_size):
                repeat_factor = attr_batch_size // static_batch_size
                if repeat_factor > 1 and attr_batch_size % static_batch_size == 0:
                    if isinstance(gt_cache, torch.Tensor):
                        gt_cache = gt_cache.repeat(repeat_factor, *([1] * (gt_cache.dim() - 1)))
                    elif isinstance(gt_cache, dict):
                        expanded_gt = {}
                        for key, value in gt_cache.items():
                            if isinstance(value, torch.Tensor):
                                expanded_gt[key] = value.repeat(repeat_factor, *([1] * (value.dim() - 1)))
                            else:
                                expanded_gt[key] = value
                        gt_cache = expanded_gt
            
            # 使用模型适配器的前向传播函数
            res = self.attr_base.model_forward_wrapper(
                attribution_inputs,
                static_inputs,
                gt_cache
            )
            return res
        
        return forward_func
    
    def _create_attribution_method(self):
        """
        创建Captum归因方法实例
        
        Returns:
            attr_method: Captum归因方法实例
        """
        method_class = self.captum_methods[self.method_name]
        
        # 处理方法特定参数
        method_kwargs = self.kwargs.copy()
        
        # 某些方法需要特殊处理
        if self.method_name in ['GuidedGradCam', 'LRP']:
            # 这些方法需要指定层
            if 'layer' not in method_kwargs:
                # 尝试获取模型的最后一层
                layers = list(self.model.named_modules())
                if layers:
                    method_kwargs['layer'] = layers[-1][1]
        
        return method_class(self.forward_func, **method_kwargs)
    
    def get_baseline(self, attribution_inputs: Dict, baseline_type: str = 'zero') -> Tuple:
        """
        使用BaselineGenerator生成Captum格式的基线
        
        Args:
            attribution_inputs: 归因输入字典
            baseline_type: 基线类型
            
        Returns:
            baselines: 基线张量元组
        """
        baselines = []
        
        # 按输入键的顺序处理
        for key in attribution_inputs.keys():
            tensor = attribution_inputs[key]
            
            # 根据输入类型确定tensor_type
            if 'traj' in key.lower():
                tensor_type = 'trajectory'
            elif 'map' in key.lower() or 'polyline' in key.lower():
                tensor_type = 'map'
            else:
                tensor_type = 'general'
            
            # 使用BaselineGenerator生成基线
            baseline = self.baseline_generator.generate_baseline(
                tensor, baseline_type, tensor_type
            )
            
            baselines.append(baseline)
        
        return tuple(baselines) if baselines else None
    
    def compute_attribution(self, attribution_inputs: Dict, static_inputs: Dict, 
                           input_tensors: Tuple) -> Dict[str, torch.Tensor]:
        """
        计算Captum归因
        
        Args:
            attribution_inputs: 归因输入字典
            static_inputs: 静态输入字典
            input_tensors: 需要计算梯度的输入张量元组
            
        Returns:
            attributions: 归因结果字典
        """
        # 缓存静态输入用于前向传播
        self._static_inputs_cache = static_inputs
        
        # 缓存输入键的顺序
        self._input_keys_cache = list(attribution_inputs.keys())
        
        # 缓存GT信息
        if 'center_gt_trajs' in static_inputs:
            self._gt_cache = static_inputs['center_gt_trajs'][..., :2]  # 只取x,y坐标
        else:
            self._gt_cache = None
        
        # 准备输入张量（按attribution_inputs的键顺序）
        ordered_inputs = tuple(attribution_inputs[key] for key in self._input_keys_cache)
        
        # 生成基线
        baselines = None
        if self.method_name in ['IntegratedGradients', 'DeepLift', 'DeepLiftShap', 
                               'GradientShap', 'ShapleyValueSampling']:
            baselines = self.get_baseline(attribution_inputs, 'zero')
        
        # 设置目标（通常为None，因为我们返回标量）
        target = None
        
        # 计算归因
        print(f"使用 {self.method_name} 计算归因...")
        if self.method_name == 'IntegratedGradients':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                baselines=baselines,
                target=target,
                n_steps=self.kwargs.get('n_steps', 50),
                method=self.kwargs.get('method', 'gausslegendre')
            )
        elif self.method_name in ['DeepLift', 'DeepLiftShap']:
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                baselines=baselines,
                target=target
            )
        elif self.method_name == 'GradientShap':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                baselines=baselines,
                target=target,
                n_samples=self.kwargs.get('n_samples', 50),
                stdevs=self.kwargs.get('stdevs', 0.0)
            )
        elif self.method_name == 'InputXGradient':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                target=target
            )
        elif self.method_name == 'Saliency':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                target=target,
                abs=self.kwargs.get('abs', True)
            )
        elif self.method_name == 'ShapleyValueSampling':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                baselines=baselines,
                target=target,
                n_samples=self.kwargs.get('n_samples', 25),
                show_progress=self.kwargs.get('show_progress', True)
            )
        elif self.method_name == 'Occlusion':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                target=target,
                sliding_window_shapes=self.kwargs.get('sliding_window_shapes', (1,) * ordered_inputs[0].dim()),
                strides=self.kwargs.get('strides', None)
            )
        else:
            # 通用调用
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                target=target,
                **{k: v for k, v in self.kwargs.items()
                   if k not in ['n_steps', 'method', 'n_samples', 'stdevs', 'abs',
                               'show_progress', 'sliding_window_shapes', 'strides']}
            )
        
        # 转换为字典格式
        attributions = {}
        if not isinstance(attrs, tuple):
            attrs = (attrs,)
        
        for i, attr in enumerate(attrs):
            if i < len(self._input_keys_cache):
                attributions[self._input_keys_cache[i]] = attr
        
        # 清除缓存
        if hasattr(self, '_static_inputs_cache'):
            del self._static_inputs_cache
        if hasattr(self, '_input_keys_cache'):
            del self._input_keys_cache
        if hasattr(self, '_gt_cache'):
            del self._gt_cache
        
        return attributions