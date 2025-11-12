
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
from utils_lc.model_dop import CN_FC
import captum.attr as capattr
from utils_attr.manyLCs_attr.class_lc_attr_basic import LCAttr


class DOPAttrCal(LCAttr):
    """
    通用的归因 AS 计算类别
    """
    def __init__(self, opt):
        super().__init__(opt)
        
        # hm_text 定义的一些变量
        self.text_stat = None
        self.text_car_names = None
        self.text_feas = None
        self.text_ten_feas = None
        self.hm_text()
        
        # DOP 模型的输入是多分支的, 所以这里要定义三个
        self.input_names = ['ego', 'sur', 'egov']
        
        # 读入 DOP 模型
        model = CN_FC(flag_soft=opt.flag_soft)
        checkpoint = torch.load(opt.net_path)
        model.load_state_dict(checkpoint)
        model = model.to(opt.device)
        model.eval()
        
        self.sur_car_num = 7
        self.forward_func = model
    
    def hm_text(self):
        if self.flag_chn_text:
            self.text_stat = ['mean', 'std', 'median', '25%', '75%', 'min', 'max']
            self.text_car_names = ['前车', '左前车', '右前车', '左后车', '右后车', '左车', '右车']
            self.text_feas = ['相对位置X', '相对位置Y', '纵向速度', '横向速度', '纵向加速度', '横向加速度', '距前车距离', '距前车时间']
            self.text_ten_feas = ['自车与前车速度差', '左前与前车速度差', '右前与前车速度差', '左前与前车距离', '右前与前车距离',
                                  '自车与左后速度差', '自车与右后速度差', '自车与左后距离', '自车与右后距离', '安全性1.5秒']
        else:
            self.text_stat = ['mean', 'std', 'median', '25%', '75%', 'min', 'max']
            self.text_car_names = ['P', 'LP', 'RP', 'LF', 'RF', 'L', 'R']
            self.text_feas = ['RelLocX', 'RelLocY', 'LongVelX', 'LatVelY', 'LongAccX', 'LatAccY', 'HeadDist', 'HeadTime']
            self.text_ten_feas = ['Ego-P-VelDiff', 'LP-P-VelDiff', 'RP-P-VelDiff', 'LP-P-Dis', 'RP-P-Dis',
                                  'Ego-LF-VelDiff', 'Ego-RF-VelDiff', 'Ego-LF-Dis', 'Ego-RF-Dis', 'Safe']
    
    def get_layer_module(self):
        layer_module = self.forward_func.conv_surround[0]
        return layer_module
    
    def load_attr_as_arr(self, metadata, input_tensors):
        attr_temp = self.load_attr(metadata, input_tensors)
        
        # 创建一个 8x7x8 的矩阵，初始化为零
        result = np.zeros((8, 7, 8))
        result[0, :, :] = attr_temp[0]
        result[1:, :, :] = attr_temp[1]
        return result
    
    def load_attr(self, metadata, input_tensors):
        # 生成决策, 并取得最大索引, 同时改变保存名字
        scores = self.forward_func(*input_tensors)
        attr_index = torch.argmax(scores, dim=1, keepdim=True)
        curr_dec = str(utils_save.from_tensor_to_np(attr_index))
        metadata['curr_dec'] = curr_dec
        
        # 处理生成一些需要的路径
        self.proc_dir(metadata)
        
        if 'attr_hm_name' in metadata:
            attr_hm_name = metadata['attr_hm_name']
        else:
            attr_hm_name = self.attr_hm_name + '-' + 'OurIG'
            
        attr_temp = []
        for i in range(len(self.input_names)):
            curr_input_name = self.input_names[i]
            attr_save_name = attr_hm_name+'-'+curr_input_name+'.npy'
            attr_save_path = self.attr_np_save_dir + '/' + attr_save_name
            curr_attr = np.load(attr_save_path)
            attr_temp.append(curr_attr)
        return attr_temp
    
    @staticmethod
    def proc_attr(**kwargs):
        # 将 attr tensor 转换为 np
        attrs_ori_l = []
        attrs_filtered_l = []
        len_attrs = len(kwargs['attrs'])
        for i in range(len_attrs):
            attrs_ori_np = utils_save.from_tensor_to_np(kwargs['attrs'][i])
            attrs_ori_l.append(attrs_ori_np)
            
            attrs_filtered = attrs_ori_np * (attrs_ori_np > np.quantile(attrs_ori_np, kwargs['quantile']))
            attrs_filtered_l.append(attrs_filtered)
        return attrs_ori_l, attrs_filtered_l
    
    def save_res(self, attrs_ori_l, attrs_filtered_l, attr_hm_name):
        if self.opt.flag_save_attr:
            self.save_hm(attrs_ori_l, self.attr_save_dir, attr_hm_name, attrs_filtered_l)
            
        if self.opt.flag_save_attr and not self.flag_attr_np_load:
            for i in range(len(self.input_names)):
                curr_input_name = self.input_names[i]
                curr_attr = attrs_ori_l[i]
                
                attr_save_name = attr_hm_name+'-'+curr_input_name+'.npy'
                attr_save_path = self.attr_np_save_dir + '/' + attr_save_name
                np.save(attr_save_path, curr_attr)
    
    def save_hm(self, input_l, save_dir, attr_hm_name, filtered_l=None):
        # 保存 tuple 3 的 input 或者 attr 为热力图
        # 把 input 信息保存为热力图
        for i in range(len(self.input_names)):
            curr_input_name = self.input_names[i]
            curr_input = input_l[i]
            curr_input_np = utils_save.from_tensor_to_np(curr_input)
            
            if filtered_l is not None:
                filtered = filtered_l[i]
                filtered_np = utils_save.from_tensor_to_np(filtered)
                
            if curr_input_np.ndim == 1:
                curr_input_np = np.expand_dims(curr_input_np, 0)
                if filtered_l is not None:
                    filtered_np = np.expand_dims(filtered_np, 0)
            
            if filtered_l is not None:
                data_for_range = filtered_np
            else:
                data_for_range = curr_input_np
                
            if self.flag_chn_text:
                save_name_temp = 'chn-' + attr_hm_name
            else:
                save_name_temp = attr_hm_name
                
            if curr_input_name == 'sur':
                for i_car in range(self.sur_car_num):
                    save_name = save_name_temp+'-'+curr_input_name+str(i_car)
                    hm_arr = curr_input_np[i_car]
                    row_labels = self.text_stat
                    col_labels = [self.text_car_names[i_car] + '-' + item for item in self.text_feas]
                    utils_attr_save.save_doplc_attr_hm(save_dir, save_name,
                                                       row_labels, col_labels,
                                                       attrs=hm_arr, attrs_filter=data_for_range,
                                                       flag_chn_text=self.flag_chn_text)
            elif curr_input_name == 'ego':
                save_name = save_name_temp+'-'+curr_input_name
                hm_arr = curr_input_np
                row_labels = self.text_stat
                col_labels = ['Ego-' + item for item in self.text_feas]
                utils_attr_save.save_doplc_attr_hm(save_dir, save_name,
                                                   row_labels, col_labels,
                                                   attrs=hm_arr, attrs_filter=data_for_range,
                                                   flag_chn_text=self.flag_chn_text)
            else:
                save_name = save_name_temp+'-'+curr_input_name
                hm_arr = curr_input_np
                row_labels = ['-']
                col_labels = self.text_ten_feas
                utils_attr_save.save_doplc_attr_hm(save_dir, save_name,
                                                   row_labels, col_labels,
                                                   attrs=hm_arr, attrs_filter=data_for_range,
                                                   flag_chn_text=self.flag_chn_text)
                
    def attr_cal(self, metadata, input_tensors):
        # 生成决策, 并取得最大索引, 同时改变保存名字
        scores = self.forward_func(*input_tensors)
        
        # attr_index = torch.argmax(scores, dim=1, keepdim=True)
        attr_index = torch.argmax(scores, dim=1, keepdim=False)
        curr_dec = str(utils_save.from_tensor_to_np(attr_index))
        metadata['curr_dec'] = curr_dec
        
        # 处理生成一些需要的路径
        self.proc_dir(metadata)
        
        # 创建一个默认值为字典的defaultdict
        attr_res = defaultdict(dict)
        
        attr_quantile = self.opt.attr_quantile
        
        # 把输入保存为 hm
        self.save_hm(input_tensors, self.input_save_dir, self.attr_hm_name)
        
        attr_methods = metadata['attr_methods']
        for attr_method_name in attr_methods:
            attr_hm_name = self.attr_hm_name + '-' + attr_method_name
            metadata['attr_hm_name'] = attr_hm_name
            attr_method_cap_name = self.attr_method_save_names[attr_method_name]
            
            if not self.flag_attr_np_load:
                # 设定归因方法
                if attr_method_cap_name in ['SPI']:
                    baselines, baseline_scales = self.get_baseline(input_tensors, flag_base='1-')
                elif attr_method_name in ['OurIG']:
                    baselines, baseline_scales = self.get_baseline(input_tensors, flag_base='0-')
                    attr_method = getattr(capattr, attr_method_cap_name)(self.forward_func)
                elif attr_method_name in ['DeepLiftShap']:
                    baselines, baseline_scales = self.get_baseline(input_tensors, flag_base='multi09')
                    attr_method = getattr(capattr, attr_method_cap_name)(self.forward_func)
                else:
                    baselines, baseline_scales = self.get_baseline(input_tensors, flag_base='00')
                    attr_method = getattr(capattr, attr_method_cap_name)(self.forward_func)
                
                if attr_method_name in ['DeepLift', 'DeepLiftShap', 'GradientShap', 'IntegratedGradients', 'OurIG']:
                    attrs = attr_method.attribute(inputs=input_tensors, target=attr_index, baselines=baselines)
                elif attr_method_name in ['InputXGradient']:
                    attrs = attr_method.attribute(inputs=input_tensors, target=attr_index)
                elif attr_method_name == 'ShapleySampling':
                    attrs = attr_method.attribute(inputs=input_tensors, target=attr_index, baselines=baselines,
                                                  n_samples=10, show_progress=False)
            else:
                attrs = self.load_attr(metadata, input_tensors)
            # 处理并保存 attr
            attrs_ori_l, attrs_filtered_l = self.proc_attr(attrs=attrs, quantile=attr_quantile)
            self.save_res(attrs_ori_l, attrs_filtered_l, attr_hm_name)
            attr_res[attr_method_name]['attrs_ori'] = attrs_ori_l
            attr_res[attr_method_name]['attrs_filtered'] = attrs_filtered_l
        return attr_res
    
    def get_baseline(self, input_tensors, flag_base):
        # 全 0 baseline
        baseline0 = torch.zeros(input_tensors[0].size(), device=self.opt.device)
        baseline1 = torch.zeros(input_tensors[1].size(), device=self.opt.device)
        baseline2 = torch.zeros(input_tensors[2].size(), device=self.opt.device)
        baselines = (baseline0, baseline1, baseline2)
        baseline_scales = (baseline0/input_tensors[0], baseline1/input_tensors[1], baseline2/input_tensors[2])
        return baselines, baseline_scales
