
import os

import utils_data.IO as IO
from utils_data.IO import create_dir, get_proj_abs_dir
import torch


class Config:
    abs_dir = get_proj_abs_dir()

    # ========================================================================
    
    # 数据 参数
    batch_size = 1
    data_idx = 4
    # car_ids = [218, 282, 736, 272, 274]
    # car_ids = [218, 736]
    car_ids = [74]
    
    # ========================================================================
    
    # attr 参数
    # 是否读取计算好的归因
    flag_attr_np_load = True
    
    # 保存归因热力图以及归因 np 文件
    flag_save_attr = True
    
    # 保存的归因热力图中文字是否使用中文
    flag_chn_text = False
    
    # 网络进行 softmax 计算不
    flag_soft = False
    
    # 是否保存归因热力图在同一文件夹
    flag_save_one_dir = False
    attr_save_root_dir = abs_dir + '/all_exps/experiments_manyLCs/dopLC_attr_res'
    
    if flag_save_one_dir:
        attr_save_dir = attr_save_root_dir
        unformat_save_dir = attr_save_root_dir + '/{}'
    else:
        attr_save_dir = attr_save_root_dir + '/{}'
    
    input_save_dir = attr_save_root_dir + '/{}/input'
    attr_np_save_dir = attr_save_root_dir + '/{}/attr_np'
    
    # 设定的是 filter 掉的归因值分位数
    attr_quantile = 0.75
    
    # attr_methods = ['OurIG', 'Dirichlet', 'DeepLift', 'DeepLiftShap', 'InputXGradient', 'ShapleySampling', 'GradientShap', 'IntegratedGradients', ]
    attr_methods = ['OurIG', 'Dirichlet']
    # with GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # net_path = abs_dir + "\\best_accuracy_net3.pth"
    # 将原来的相对路径改为绝对路径
    net_path = "D:/Autonomous-Driving/TrajAttrPub/weights_lc/weights_dop/best_accuracy_net3.pth"

    # ========================================================================
    
