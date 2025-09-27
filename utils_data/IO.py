
import os

import numpy as np
from struct import pack, unpack
import scipy.io as sio
import struct
import glob
import pathlib
import re


def get_proj_abs_dir():
    # e.g., '\\PycharmProjects\\AttrInv'
    # 输入网络的pt数据保存dir
    abs_dir = str(pathlib.Path().resolve())
    if abs_dir.endswith('AttrInv'):
        return abs_dir
    else:
        indices = re.finditer('AttrInv', abs_dir)
        for i in indices:
            span_range = i.span()
            abs_dir = abs_dir[0:span_range[1]]
            break
        return abs_dir


def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        
def save_np_arr(data_dir, save_name, arr, np_type=np.float32):
    save_path = data_dir + '/' + save_name
    arr_save = arr.astype(np_type)
    np.save(save_path, arr_save)


def load_np_arr(data_dir, save_name):
    data_path = data_dir + '/' + save_name
    res = np.load(data_path)
    return res


def bdd_data_dir():
    return "/media/shi/Data/dataset/bdd100k"


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1])
