"""
轨迹预测归因计算基础类

该类提供轨迹预测模型归因计算的统一接口，支持多种模型和归因方法
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import re

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore
from collections import defaultdict
import utils_data.utils_save as utils_save
import utils_data.IO as IO
from .distance_metrics import DistanceMetrics
from ..adapters import ModelAdapterFactory
from utils.path_manager import path_manager


class TrajAttrBase:
    """
    轨迹预测归因计算基础类
    """
    
    def __init__(self, model, config, save_paths=None):
        """
        初始化归因计算器
        
        Args:
            model: 轨迹预测模型
            config: 配置对象，包含模型和归因参数
            save_paths: 保存路径字典（由 compute_traj_attr.py 传入的 self.paths）
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # 设置随机种子
        torch.manual_seed(config.get('seed', 42))
        
        # 归因方法映射
        if hasattr(config, 'attribution'):
            self.attr_methods = config.attribution.get('methods', ['IntegratedGradients'])
        else:
            self.attr_methods = config.get('attr_methods', ['IntegratedGradients'])
        
        # 距离度量器
        self.distance_metrics = DistanceMetrics()
        
        # 保存路径设置
        self._setup_save_paths(save_paths)
        
        # 模型输入输出维度信息
        self._setup_model_info(config)
        
        # 创建模型适配器
        # 初始化模型名称为None
        model_name = None
        # 检查config对象是否有method属性，如果有则获取method中的model_name
        if hasattr(config, 'method'):
            model_name = config.method.get('model_name')
        # 如果model_name为空（即之前未获取到值），则尝试从config中直接获取model_name
        if not model_name:
            model_name = config.get('model_name')
        # 使用ModelAdapterFactory创建模型适配器，传入model和model_name
        self.model_adapter = ModelAdapterFactory.create_adapter(model, model_name)
        
    def _setup_save_paths(self, save_paths=None):
        """设置保存路径"""
        # 从配置获取保存设置
        if hasattr(self.config, 'attribution'):
            self.save_results = self.config.attribution.get('save_raw_attributions', True)
        else:
            self.save_results = self.config.get('save_attr_results', True)
            
        self.exp_name = self.config.get('exp_name', 'traj_attr')
        
        # 使用外部传入的路径字典（来自 compute_traj_attr.py 的 self.paths）
        if save_paths:
            self.save_paths = save_paths
            # 使用 numpy 目录作为主要保存目录
            self.attr_save_dir = str(save_paths.get('numpy', save_paths.get('attributions', '')))
        else:
            # 向后兼容：如果没有传入路径字典，使用默认路径
            default_path = path_manager.get_exp_res_path("res_trajattr", "default")
            self.save_paths = {'numpy': default_path}
            self.attr_save_dir = str(default_path)
        
    def _setup_model_info(self, config):
        """设置模型信息"""
        # 支持DictConfig和普通dict两种格式
        if hasattr(config, 'method'):
            self.model_name = config.method.get('model_name', config.get('model_name', 'unknown'))
        else:
            self.model_name = config.get('model_name', 'unknown')
        self.future_len = config.get('future_len', 60)
        self.past_len = config.get('past_len', 21)
        
        # 从method配置中获取模型特定信息
        if hasattr(config, 'method'):
            self.num_modes = config.method.get('num_modes', 6)
        else:
            self.num_modes = config.get('num_modes', 6)
        
    def prepare_model_for_attribution(self, batch: Dict) -> Tuple[Dict, Dict]:
        """
        为归因计算准备模型输入
        
        Args:
            batch: 原始批量数据
            
        Returns:
            attribution_inputs: 需要计算归因的输入张量字典
            static_inputs: 不需要梯度的静态输入字典
        """
        # 使用模型适配器提取输入
        attribution_inputs = self.model_adapter.get_attribution_inputs(batch)
        static_inputs = self.model_adapter.get_static_inputs(batch)
        
        return attribution_inputs, static_inputs
        
    def model_forward_wrapper(self, attribution_inputs: Dict, static_inputs: Dict, 
                              target_trajs: torch.Tensor = None) -> torch.Tensor:
        """
        模型前向传播包装器，返回用于计算归因的标量损失
        
        Args:
            attribution_inputs: 需要归因的输入张量字典
            static_inputs: 静态输入字典
            target_trajs: 目标轨迹（可选）
            
        Returns:
            scalar_loss: 标量损失用于反向传播
        """
        return self.model_adapter.forward_with_loss(
            attribution_inputs, static_inputs, target_trajs
        )
    
    def compute_attribution(self, batch: Dict, method: str = 'IntegratedGradients', 
                            **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算输入归因值
        
        Args:
            batch: 输入批量数据
            method: 归因方法名称
            **kwargs: 方法特定参数（会覆盖配置文件参数）
            
        Returns:
            attributions: 归因结果字典
        """
        # 准备输入
        attribution_inputs, static_inputs = self.prepare_model_for_attribution(batch)
        input_tensors = tuple(attribution_inputs[key] for key in attribution_inputs.keys() 
                             if hasattr(attribution_inputs[key], 'requires_grad') and 
                             attribution_inputs[key].requires_grad)
        
        # 根据方法计算归因
        
        if method == 'GuidedIG' or method == 'Guided-IG':
            from ..methods.guided_ig_attr import GuidedIGAttribution
            # 从配置中提取GuidedIG特定参数
            method_config = self.config.get('guided_ig_config', {}).copy()
            method_config.update(kwargs)  # kwargs覆盖配置文件参数
            attr_calculator = GuidedIGAttribution(self, **method_config)
            attributions = attr_calculator.compute_attribution(
                attribution_inputs, static_inputs, input_tensors)

        elif method == 'CP-AttnLRP':
            # AttnLRP with CP attention rule
            method_config = self._get_attnlrp_method_config(method)
            method_config.update(kwargs)
            method_config['attention_rule'] = 'CP'
            from ..methods.attn_lrp_attr import AttnLRPAttribution
            attr_calculator = AttnLRPAttribution(self, **method_config)
            attributions = attr_calculator.compute_attribution(
                attribution_inputs, static_inputs, input_tensors)

        elif method == 'AttnLRP':
            # 使用 AttnLRP（模块级 LRP/CP‑LRP 注意力）。
            # 从配置中提取 AttnLRP 专属参数（可选）。
            method_config = self._get_attnlrp_method_config(method)
            method_config.update(kwargs)
            from ..methods.attn_lrp_attr import AttnLRPAttribution
            attr_calculator = AttnLRPAttribution(self, **method_config)
            attributions = attr_calculator.compute_attribution(
                attribution_inputs, static_inputs, input_tensors)

        else:
            # 使用Captum方法
            from ..methods.captum_attr import CaptumAttribution
            # 从配置中提取Captum特定参数
            method_config = self._get_captum_method_config(method)
            method_config.update(kwargs)  # kwargs覆盖配置文件参数
            # 检查配置中是否有attribution属性，如果有则从中获取distance_type，否则直接从config中获取
            # 默认值为'min_ade'
            if hasattr(self.config, 'attribution'):
                distance_type = self.config.attribution.get('distance_type', 'min_ade')
            else:
                distance_type = self.config.get('distance_type', 'min_ade')
            # 创建CaptumAttribution实例，传入self、method、distance_type和method_config
            attr_calculator = CaptumAttribution(
                self,
                method,
                distance_type=distance_type,
                **method_config
            )
            # 计算归因，传入attribution_inputs、static_inputs和input_tensors
            attributions = attr_calculator.compute_attribution(
                attribution_inputs, static_inputs, input_tensors)

        return attributions

    def _get_captum_method_config(self, method_name: str) -> Dict[str, Any]:
        """从配置中提取某个Captum方法的专属参数"""
        captum_cfg = self.config.get('captum_config', {})

        if isinstance(captum_cfg, DictConfig) and OmegaConf is not None:
            captum_cfg = OmegaConf.to_container(captum_cfg, resolve=True)

        if not isinstance(captum_cfg, dict):
            return {}

        method_key = self._normalize_method_key(method_name)

        if method_key in captum_cfg and isinstance(captum_cfg[method_key], dict):
            return captum_cfg[method_key].copy()

        if method_name in captum_cfg and isinstance(captum_cfg[method_name], dict):
            return captum_cfg[method_name].copy()

        # 回退：仅保留非字典类型的通用参数
        return {k: v for k, v in captum_cfg.items() if not isinstance(v, dict)}

    def _normalize_method_key(self, name: str) -> str:
        """将方法名转换为配置使用的snake_case键"""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    def save_attribution_results(self, attributions: Dict, batch: Dict, 
                                 method: str, metadata: Optional[Dict] = None):
        """
        保存归因结果
        
        Args:
            attributions: 归因结果
            batch: 原始批量数据  
            method: 归因方法名称
            metadata: 额外的元数据
        """
        if not self.save_results:
            return
            
        # 使用传入的路径字典中的 numpy 目录（已由 compute_traj_attr.py 创建）
        # Save under method-named subdirectory inside numpy dir
        method_dir = os.path.join(self.attr_save_dir, str(method))
        os.makedirs(method_dir, exist_ok=True)
        np_save_dir = method_dir
        
        batch_size = list(attributions.values())[0].size(0)
        batch_id = metadata.get('batch_id') if metadata else None
        scenario_ids = metadata.get('scenario_ids', []) if metadata else []
        file_prefixes = [] if metadata is None else metadata.setdefault('file_prefixes', [None] * batch_size)

        for batch_idx in range(batch_size):
            if metadata is not None:
                prefix = file_prefixes[batch_idx]
            else:
                prefix = None

            if not prefix:
                if batch_idx < len(scenario_ids):
                    prefix = f"scene_{scenario_ids[batch_idx]}"
                else:
                    base = batch_id if batch_id is not None else batch_idx
                    prefix = f"batch_{base}"
                    if batch_size > 1:
                        prefix += f"_{batch_idx}"

                if metadata is not None:
                    file_prefixes[batch_idx] = prefix

            save_name = f"{prefix}_{method}"

            # 保存numpy数组
            for input_name, attr_values in attributions.items():
                attr_np = utils_save.from_tensor_to_np(attr_values[batch_idx])
                np_path = os.path.join(np_save_dir, f"{save_name}_{input_name}.npy")
                np.save(np_path, attr_np)
                
        print(f"归因结果已保存至: {self.attr_save_dir}")

    # ---- AttnLRP helper config --------------------------------------------
    def _get_attnlrp_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        获取注意力LRP（Layer-wise Relevance Propagation）方法的配置信息
        参数:
            method_name (str): 方法名称，虽然当前函数未使用此参数，但保留以备将来扩展
        返回:
            Dict[str, Any]: 包含注意力LRP配置的字典，如果配置不存在则返回空字典
        """
    # 从配置中获取默认的注意力LRP配置，如果不存在则为空字典
        attnlrp_cfg = self.config.get('attn_lrp_config', {})
    # 检查配置对象是否有attribution属性
        if hasattr(self.config, 'attribution'):
            attribution_cfg = self.config.attribution
        # 如果attribution配置中有attn_lrp_config，则使用它覆盖默认配置
            if hasattr(attribution_cfg, 'attn_lrp_config'):
                attnlrp_cfg = attribution_cfg.attn_lrp_config
    # 尝试将配置对象转换为字典
        if hasattr(attnlrp_cfg, 'to_dict'):
            attnlrp_cfg = attnlrp_cfg.to_dict()
    # 如果对象没有to_dict方法但有_content属性，则使用_content
        elif hasattr(attnlrp_cfg, '_content'):
            attnlrp_cfg = attnlrp_cfg._content
    # 确保最终返回的是字典类型，如果不是则返回空字典
        if not isinstance(attnlrp_cfg, dict):
            return {}
    # 返回配置的副本以避免外部修改
        return attnlrp_cfg.copy()

    def compute_and_save_attribution(self, batch: Dict, methods: List[str] = None,
                                     metadata: Optional[Dict] = None) -> Dict:
        """
        计算并保存归因结果的便利函数
        
        Args:
            batch: 输入批量数据
            methods: 归因方法列表
            metadata: 额外元数据
            
        Returns:
            all_attributions: 所有方法的归因结果
        """
        if methods is None:
            methods = self.attr_methods
            
        all_attributions = {}
        
        for method in methods:
            print(f"计算 {method} 归因...")
            attributions = self.compute_attribution(batch, method)
            all_attributions[method] = attributions
            
            # 保存结果
            self.save_attribution_results(attributions, batch, method, metadata)
        return all_attributions
