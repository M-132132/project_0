
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
                if attr_method_cap_name in ['SPI', 'Dirichlet']:
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
                elif attr_method_name == 'Dirichlet':
                    attrs = self.cal_Dirichlet_attr(inputs=input_tensors, baseline=baselines, baseline_scales=baseline_scales,
                                                    target=attr_index, alpha=0.04, n_paths=35, n_steps=30)
            else:
                attrs = self.load_attr(metadata, input_tensors)
            # 处理并保存 attr
            attrs_ori_l, attrs_filtered_l = self.proc_attr(attrs=attrs, quantile=attr_quantile)
            self.save_res(attrs_ori_l, attrs_filtered_l, attr_hm_name)
            attr_res[attr_method_name]['attrs_ori'] = attrs_ori_l
            attr_res[attr_method_name]['attrs_filtered'] = attrs_filtered_l
        return attr_res
    
    def cal_Dirichlet_attr(self, inputs, baseline, baseline_scales, target, alpha=0.04, n_paths=30, n_steps=30):
        torch.manual_seed(3)
        # inputs 包含了输入的 3 个 tensors, 所以后面输入都需要处理 3 个 tensors
        # 其他指数分布族 (缩放 sample 结果至 0-1) 也有类似效果, 代码可以参考 check_torch_distrib,
        concentration = alpha * torch.ones(n_steps-1)  # 可以调整concentration参数改变随机性
        dirichlet_distrib = torch.distributions.Dirichlet(concentration)
        
        inputs_shapes = []
        for i_input in range(3):
            inputs_shape = inputs[i_input].shape
            inputs_shapes.append(inputs_shape)
            
        linspace_values = []
        for i_input in range(3):
            linspace_value = torch.linspace(0, 1, n_steps, device=self.opt.device)
            linspace_value = linspace_value.reshape([-1, ] + [1, ] * len(inputs_shapes[i_input]))
            linspace_values.append(linspace_value)
            
        # -----------------------------------------------------------------------------------
        rand_igs = [[], [], []]
        for i in range(n_paths):
            rand_xpaths = []
            for i_input in range(3):
                # 采样出 Dirichlet 数据点, 并进行累加
                # 这里 gaps 是 1*1*7*8*29, 1 是 batch size, 1 是一个车, 7*8 是特征矩阵, 29 是步数
                gaps = dirichlet_distrib.sample([*inputs_shapes[i_input]]).to(self.opt.device)
                sequences = torch.zeros([*inputs_shapes[i_input], n_steps], device=self.opt.device)
                sequences[..., 1:] = torch.cumsum(gaps, dim=-1)
                if len(inputs_shapes[i_input]) == 4:
                    sequences = sequences.permute(4, 0, 1, 2, 3)
                else:
                    sequences = sequences.permute(2, 0, 1)
                
                # 过滤部分结果
                u = torch.rand_like(sequences) < linspace_values[i_input]
                u_np = utils_save.from_tensor_to_np(u)
                sequence_rand = torch.mul(sequences, u)
                
                sequence_rand_np = utils_save.from_tensor_to_np(sequence_rand)
                
                # 与 inputs 相乘构建分段路径
                rand_xpath_temp = (torch.mul(sequence_rand, inputs[i_input][None, ]) +
                                   torch.mul(1-sequence_rand, baseline[i_input][None, ])).requires_grad_(True)
                rand_xpaths.append(rand_xpath_temp)
                
            # 沿着路径收集梯度
            rand_gpaths0 = []
            rand_gpaths1 = []
            rand_gpaths2 = []
            for tensor1, tensor2, tensor3 in zip(*rand_xpaths):
                _x = (tensor1, tensor2, tensor3)
                _y_temp = self.forward_func(*_x)
                _y = _y_temp[0, target]
                _g = torch.autograd.grad(_y, _x)
                rand_gpaths0.append(_g[0])
                rand_gpaths1.append(_g[1])
                rand_gpaths2.append(_g[2])
            
            rand_gpaths = [rand_gpaths0, rand_gpaths1, rand_gpaths2]
            for i_input in range(3):
                rand_gpath = rand_gpaths[i_input]
                rand_gpath = torch.stack(rand_gpath)
                # 使用黎曼积分收集梯度
                rand_xpath = rand_xpaths[i_input]
                rand_ig = torch.mul(rand_xpath[1:]-rand_xpath[:-1], (rand_gpath[1:]+rand_gpath[:-1])/2).sum(dim=0)
                rand_igs[i_input].append(rand_ig)
        
        attr_res = []
        for i_input in range(3):
            attr_res.append(torch.mean(torch.stack(rand_igs[i_input]), dim=0))
        return attr_res
    
    def get_baseline(self, input_tensors, flag_base):
        # 全 0 baseline
        baseline0 = torch.zeros(input_tensors[0].size(), device=self.opt.device)
        baseline1 = torch.zeros(input_tensors[1].size(), device=self.opt.device)
        baseline2 = torch.zeros(input_tensors[2].size(), device=self.opt.device)
        baselines = (baseline0, baseline1, baseline2)
        baseline_scales = (baseline0/input_tensors[0], baseline1/input_tensors[1], baseline2/input_tensors[2])
        return baselines, baseline_scales
