import glob
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch


def find_latest_checkpoint(search_pattern):

    # List all files matching the pattern
    list_of_files = glob.glob(search_pattern, recursive=True)
    # Find the file with the latest modification time
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch Lightning based training.

    该函数用于设置随机种子以确保实验的可重复性，适用于基于PyTorch Lightning的训练过程。
    它会设置PyTorch、CUDA（如果使用）、numpy、Python的random模块以及PyTorch Lightning的随机种子。
    Args:
    seed_value (int): The seed value to be set for random number generators.
                     默认值为42，用于设置各种随机数生成器的种子值。
    """
    # Set the random seed for PyTorch
    # 为PyTorch设置随机种子，确保PyTorch的随机操作可重复
    torch.manual_seed(seed_value)

    # If using CUDA (PyTorch with GPU)

    # 如果使用CUDA（即GPU加速的PyTorch），设置以下随机种子
    torch.cuda.manual_seed(seed_value)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    # 如果使用多GPU，设置所有GPU的随机种子
    # Set the random seed for numpy (if using numpy in the project)
    # 为numpy设置随机种子（如果项目中使用了numpy）
    np.random.seed(seed_value)

    # Set the random seed for Python's `random`
    # 为Python的random模块设置随机种
    random.seed(seed_value)

    # Set the seed for PyTorch Lightning's internal operations
    # 为PyTorch Lightning的内部操作设置随机种子
    # workers=True参数确保所有数据加载工作进程的随机性也被控制
    pl.seed_everything(seed_value, workers=True)
