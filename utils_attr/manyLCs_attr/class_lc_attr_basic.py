
import os
import sys
import random
from collections import defaultdict

import numpy as np

from skimage.transform import resize

import utils_data.IO as IO
import utils_data.utils_save as utils_save
import utils_data.utils_attr_save as utils_attr_save

from typing import Any, Callable, cast, Dict, List, overload, Tuple, Union

import torch


class LCAttr:
    """
    通用的换道归因计算类
    """
    def __init__(self, opt):
        torch.manual_seed(3)
        
        # 把 opt 定义上
        self.opt = opt
        
        # 初始化归因方法对应的名字
        self.attr_method_save_names = None
        self.attr_name_method_corr()
        
        # 是否在画热力图时使用中文, 便于调试和理解
        self.flag_chn_text = opt.flag_chn_text
        
        # 保存归因信息的文件夹
        # 基本保存思路是每个数据一个文件夹, 里面分别有保存 input 和 保存归因 np 文件的子文件夹
        self.abs_dir = opt.abs_dir
        if hasattr(opt, 'attr_save_dir'):
            self.attr_save_root_dir = opt.attr_save_root_dir
            self.attr_save_dir_unformat = opt.attr_save_dir
            
            if hasattr(opt, 'input_save_dir'):
                # ori input 保存路径
                self.input_save_dir_unformat = opt.input_save_dir
            else:
                self.input_save_dir_unformat = self.attr_save_dir_unformat
            self.input_save_dir = None
            
            # attr_save_dir 是保存 attr hm 的路径, 需要根据数据名来创建
            self.attr_save_dir = None
            # file_name 是当前数据名, 用于命名文件夹; attr_hm_name 是热力图名
            self.file_name = None
            self.attr_hm_name = None
            
        if hasattr(opt, 'attr_np_save_dir'):
            self.attr_np_save_dir_unformat = opt.attr_np_save_dir
            self.flag_attr_np_load = opt.flag_attr_np_load
            
            # attr_np_save_dir 是保存 attr np 的路径, 需要根据数据名来创建
            self.attr_np_save_dir = None
        
        # 这些参数会在子类定义
        self.sur_car_num = None
        self.forward_func = None
        self.gradient_func = self.compute_decision_net_gradients
        
    def hm_text(self):
        """
        在子类中使用这个函数, 定义热力图里面的文字
        """
        pass
    
    def load_attr(self, metadata, inputs):
        """
        可能在子类定义读取计算的 attr np 文件
        """
        pass
    
    def get_layer_module(self):
        """
        可能在子类定义获取模型的特定层
        """
        pass
    
    @staticmethod
    def proc_attr(attrs, quantile):
        """
        每种模型的输入都不同相同, 所以归因的处理模型需要在子类单独定义
        """
        pass
    
    def save_res(self, **kwargs):
        """
        每种模型的热力图都不同相同, 所以用于保存的代码需要单独定义
        """
        pass
    
    def save_hm(self, **kwargs):
        """
        每种模型的热力图都不同相同, 所以用于保存的代码需要单独定义
        """
        pass
    
    def get_baseline(self, **kwargs):
        """
        每种模型的输入结构不同相同, 所以获取 baseline 的代码需要单独定义
        """
        pass
    
    @staticmethod
    def compute_decision_net_gradients(forward_fn, inputs, target):
        with torch.autograd.set_grad_enabled(True):
            # output 也需要 grad
            output = forward_fn(inputs)[0]
            grads = torch.autograd.grad(output, inputs)[0]
        return grads
    
    def attr_name_method_corr(self):
        self.attr_method_save_names = {
            'DeepLift': 'DeepLift',
            'DeepLiftShap': 'DeepLiftShap',
            'InputXGradient': 'InputXGradient',
            'ShapleySampling': 'ShapleyValueSampling',
            'ShapleySamplingZero': 'ShapleyValueSampling',
            'GradientShap': 'GradientShap',
            'IntegratedGradients': 'IntegratedGradients',
            
            'GradientShap2': 'GradientShap2',
            'SPI': 'SPI',
            'Dirichlet': 'Dirichlet',
            'OurIG': 'IntegratedGradients'
            }
    
    def proc_dir(self, metadata):
        """
        这部分代码是对应着 highD 数据集的
        """
        data_info = metadata['other_info']
        # 保存名 数据编号-ID-frame
        csv_data_idx_str = str(self.opt.data_idx).zfill(2)
        self.attr_hm_name = csv_data_idx_str + '-ID' + str(data_info[0]) + '-D' + metadata['curr_dec'] +\
                            '-frm' + str(data_info[1]) + '-' + str(data_info[2])
        
        # 由数据 idx, 和当前车辆 ID 组成, 但是不含有 frm 信息
        self.file_name = csv_data_idx_str + '-ID' + str(data_info[0])
        
        self.attr_np_save_dir = self.attr_np_save_dir_unformat.format(self.file_name)
        IO.create_dir(self.attr_np_save_dir)
        
        if self.opt.flag_save_one_dir:
            self.attr_save_dir = self.attr_save_dir_unformat
        else:
            self.attr_save_dir = self.attr_save_dir_unformat.format(self.file_name)
        IO.create_dir(self.attr_save_dir)
        
        self.input_save_dir = self.input_save_dir_unformat.format(self.file_name)
        IO.create_dir(self.input_save_dir)
        
    def grad_cal(self, inputs):
        # 生成决策, 并取得最大索引, 进而计算 grad
        with torch.autograd.set_grad_enabled(True):
            scores = self.forward_func(*inputs)
            attr_index = torch.argmax(scores, dim=1, keepdim=True)
            # grads = torch.autograd.grad(torch.unbind(scores[attr_index]), inputs)
            grads = torch.autograd.grad(torch.gather(scores, dim=1, index=attr_index), inputs)
        return grads
    
    @staticmethod
    def scale_to_range(input_tensor, min_val=0.95, max_val=1.0, eps=1e-8):
        """
        把输入 tensor 的每一完整段都缩放到指定范围
        input_tensor 尺寸是 30*50, 30 表示分为 30 段, 从 0->1
        50 表示此时 batch size 是 50, 具体就是 highD 数据的 50 frames
        """
        # 获取每组的最小值和最大值
        min_vals = input_tensor.min(dim=1, keepdim=True)[0]  # [30, 1]
        max_vals = input_tensor.max(dim=1, keepdim=True)[0]  # [30, 1]
        
        # 检查是否所有值都相同(包括全0的情况)
        is_constant = (max_vals - min_vals) < eps
        
        # 创建一个掩码来处理除法
        safe_denominator = (max_vals - min_vals).clone()
        safe_denominator[is_constant] = 1.0  # 避免除以0
        
        # 归一化到[0,1]
        normalized = (input_tensor - min_vals) / safe_denominator
        
        # 对于常数序列（包括全0序列），将normalized设置为0.5
        normalized = torch.where(is_constant.expand_as(input_tensor), 0.5, normalized)
        
        # 缩放到[min_val, max_val]
        scaled = normalized * (max_val - min_val) + min_val
        return scaled
