"""
Captum归因方法集成

将Captum库中的归因方法适配到轨迹预测模型中

# @comment-style: inline
"""

import re  # 导入正则表达式模块
import torch  # 导入PyTorch深度学习框架
from typing import Dict, Tuple, Any, Optional  # 导入类型提示相关的模块
import captum.attr as capattr  # 导入Captum的归因模块
from ..utils_traj_attr.baseline_generator import BaselineGenerator  # 导入基线生成器


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
        # 初始化基础属性
        self.attr_base = attr_base  # 存储基础属性对象
        self.model = attr_base.model  # 获取并存储模型
        self.device = attr_base.device  # 获取并存储设备信息
        self.method_name = method_name  # 存储方法名称
        self.distance_type = distance_type  # 存储距离类型
        # 选择并存储方法相关的关键字参数
        self.attribute_kwargs = self._select_method_kwargs(kwargs)
        
        # 获取距离计算函数
        self.distance_fn = attr_base.distance_metrics.get_distance_function(distance_type)
        
        # 初始化基线生成器
        self.baseline_generator = BaselineGenerator(device=self.device)
        
        # 支持的Captum方法映射
        self.captum_methods = {
            'IntegratedGradients': capattr.IntegratedGradients,  # 积分梯度方法
            'DeepLift': capattr.DeepLift,  # DeepLift方法
            'DeepLiftShap': capattr.DeepLiftShap,  # DeepLift SHAP方法
            'GradientShap': capattr.GradientShap,  # 梯度SHAP方法
            'InputXGradient': capattr.InputXGradient,  # 输入乘梯度方法
            'Saliency': capattr.Saliency,  # 显著性方法
            'GuidedBackprop': capattr.GuidedBackprop,  # 引导反向传播方法
            'GuidedGradCam': capattr.GuidedGradCam,  # 引导GradCAM方法
            'LRP': capattr.LRP,  # 层次相关性传播方法
            'ShapleyValueSampling': capattr.ShapleyValueSampling,  # Shapley值采样方法
            'FeaturePermutation': capattr.FeaturePermutation,  # 特征置换方法
            'Occlusion': capattr.Occlusion,  # 遮挡方法
            'KernelShap': capattr.KernelShap,  # 核心SHAP方法
            'Lime': capattr.Lime  # LIME方法
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
                

                for key, tensor in static_inputs.items():# 获取static_inputs的原始batch_size
                    if isinstance(tensor, torch.Tensor):
                        static_batch_size = tensor.size(0)
                        break
                
                # 如果batch_size不匹配（通常是IntegratedGradients等方法导致的）
                if (attr_batch_size is not None and static_batch_size is not None and 
                    attr_batch_size != static_batch_size):
                    
                    """
                    检查attribution_inputs和static_inputs的batch_size是否不同
                    如果两者都存在且不相等，则需要进行处理
                    """
                    # 计算重复倍数
                    repeat_factor = attr_batch_size // static_batch_size
                    if repeat_factor > 1 and attr_batch_size % static_batch_size == 0:
                        """
                        检查是否可以扩展static_inputs以匹配attribution_inputs的batch_size
                        条件：
                        1. 重复倍数大于1
                        2. attr_batch_size能被static_batch_size整除
                        """
                        # 扩展static_inputs以匹配attribution_inputs的batch_size
                        expanded_static_inputs = {}
                        for key, value in static_inputs.items():
                            if isinstance(value, torch.Tensor):
                                """
                                检查当前值是否为torch.Tensor类型
                                如果是，则沿batch维度进行重复扩展
                                """
                                # 沿batch维度重复
                                expanded_value = value.repeat(repeat_factor, *([1] * (value.dim() - 1)))
                                expanded_static_inputs[key] = expanded_value
                            else:
                                # 如果不是Tensor类型，直接保持原值
                                expanded_static_inputs[key] = value
                        # 更新static_inputs为扩展后的值
                        static_inputs = expanded_static_inputs
            
            # 处理GT数据的扩展
            # 获取_gt_cache属性，如果不存在则为None
            gt_cache = getattr(self, '_gt_cache', None)
            # 检查gt_cache是否存在，以及attr_batch_size和static_batch_size是否都不为None且不相等
            if (gt_cache is not None and attr_batch_size is not None and static_batch_size is not None and
                attr_batch_size != static_batch_size):
                # 计算重复因子，即attr_batch_size与static_batch_size的比值
                repeat_factor = attr_batch_size // static_batch_size
                # 检查重复因子是否大于1，且attr_batch_size是否能被static_batch_size整除
                if repeat_factor > 1 and attr_batch_size % static_batch_size == 0:
                    # 如果gt_cache是torch.Tensor类型，则对其进行维度扩展
                    if isinstance(gt_cache, torch.Tensor):
                        # 使用repeat方法扩展tensor的第一维，其他维度保持不变
                        gt_cache = gt_cache.repeat(repeat_factor, *([1] * (gt_cache.dim() - 1)))
                    # 如果gt_cache是字典类型，则对字典中的每个tensor值进行扩展
                    elif isinstance(gt_cache, dict):
                        expanded_gt = {}
                        # 遍历字典中的每个键值对
                        for key, value in gt_cache.items():
                            # 如果值是tensor类型，则进行维度扩展
                            if isinstance(value, torch.Tensor):
                                expanded_gt[key] = value.repeat(repeat_factor, *([1] * (value.dim() - 1)))
                            # 非tensor类型保持不变
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
        method_kwargs = self.attribute_kwargs.copy()

        # 某些方法需要特殊处理
        if self.method_name in ['GuidedGradCam', 'LRP']:
            # 这些方法需要指定层
            layer = method_kwargs.pop('layer', None)
            if layer is None:
                # 尝试获取模型的最后一层
                layers = list(self.model.named_modules())
                if layers:
                    layer = layers[-1][1]
            if layer is not None:
                constructor_kwargs = {'layer': layer}
            else:
                constructor_kwargs = {}
        else:
            constructor_kwargs = {}

        # 更新剩余参数供 attribute 调用使用
        self.attribute_kwargs = method_kwargs.copy()

        return method_class(self.forward_func, **constructor_kwargs)
    
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
        # 遍历attribution_inputs字典中的所有键
        for key in attribution_inputs.keys():
            # 获取当前键对应的张量值
            tensor = attribution_inputs[key]
            
            # 根据输入类型确定tensor_type
            # 根据键名(key)判断张量(tensor)的类型
            if 'traj' in key.lower():  # 如果键名中包含'traj'(不区分大小写)
                tensor_type = 'trajectory'  # 则将张量类型设置为'trajectory'(轨迹)
            elif 'map' in key.lower() or 'polyline' in key.lower():  # 如果键名中包含'map'或'polyline'(不区分大小写)
                tensor_type = 'map'  # 则将张量类型设置为'map'(地图)
            else:  # 其他情况
                tensor_type = 'general'  # 默认将张量类型设置为'general'(通用)
            
            # 使用BaselineGenerator生成基线
            baseline = self.baseline_generator.generate_baseline(
                tensor, baseline_type, tensor_type
            )
            
            baselines.append(baseline)
        
        return tuple(baselines) if baselines else None
    
    def _select_method_kwargs(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """提取与当前方法对应的配置参数"""
        if not isinstance(config_dict, dict):
            return {}

# 获取规范化后的方法名作为键
        method_key = self._normalize_method_key(self.method_name)

# 检查规范化后的方法名是否存在于配置字典中，并且对应的值是否为字典类型
# 如果满足条件，则返回该配置字典的深拷贝
        if method_key in config_dict and isinstance(config_dict[method_key], dict):
            return config_dict[method_key].copy()

# 如果规范化后的方法名不存在，则检查原始方法名是否存在于配置字典中，并且对应的值是否为字典类型
# 如果满足条件，则返回该配置字典的深拷贝
        if self.method_name in config_dict and isinstance(config_dict[self.method_name], dict):
            return config_dict[self.method_name].copy()

        # 回退：保留非字典型的全局参数（避免把其他方法的嵌套配置一起带上）
        return {k: v for k, v in config_dict.items() if not isinstance(v, dict)}

    def _normalize_method_key(self, name: str) -> str:
        """将方法名转换为配置中使用的 snake_case 键"""
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return snake

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
        method_args = self.attribute_kwargs
        if self.method_name == 'IntegratedGradients':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                baselines=baselines,
                target=target,
                n_steps=method_args.get('n_steps', 50),
                method=method_args.get('method', 'gausslegendre')
            )
        # 判断当前使用的方法是否为DeepLift或DeepLiftShap
        elif self.method_name in ['DeepLift', 'DeepLiftShap']:
            # 使用属性归因方法计算特征重要性
            attrs = self.attr_method.attribute(
                # 输入数据，已经过排序
                inputs=ordered_inputs,
                # 基线数据，用于对比
                baselines=baselines,
                # 目标类别或标签
                target=target
            )
        elif self.method_name == 'GradientShap':# 调用属性计算方法获取特征重要性
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,        # 输入数据，已按顺序排列
                baselines=baselines,          # 基线数据，用于对比
                target=target,                # 目标类别或索引
                n_samples=method_args.get('n_samples', 50),  # 采样数量，默认为50
                stdevs=method_args.get('stdevs', 0.0)        # 标准差，默认为0.0
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
                abs=method_args.get('abs', True)
            )
        elif self.method_name == 'ShapleyValueSampling':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                baselines=baselines,
                target=target,
                n_samples=method_args.get('n_samples', 25),
                show_progress=method_args.get('show_progress', True)
            )
        elif self.method_name == 'Occlusion':
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                target=target,
                sliding_window_shapes=method_args.get('sliding_window_shapes', (1,) * ordered_inputs[0].dim()),
                strides=method_args.get('strides', None)
            )
        else:
            # 通用调用
            attrs = self.attr_method.attribute(
                inputs=ordered_inputs,
                target=target,
                **{k: v for k, v in method_args.items()
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
