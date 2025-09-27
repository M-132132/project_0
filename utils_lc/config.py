
# utils_lc/config.py
from utils.path_manager import path_manager

# 使用路径管理器获取数据集根目录
DATASET_ROOT = path_manager.get_root_path("dataset_lc")

# frame rate 25
FEATURE_CHOICE = "CNN_FC"  # "CNN_FC" for DOP, "Normal" for LSTM
LANE_CHANGE_KEEP_RATIO = 3
FRAME_TAKEN = 50  # number of states to construct features, 50 means 2 seconds
FRAME_BEFORE = 25  # frame taken before the lane change, 25 means 1 second
FRAME_CONSIST = 300  # frame requirement before the lane keep, 300 means 12 seconds
FRAME_BEFORE_FLAG = False  # use FRAME_BEFORE or not
SAFETIME_HEADWAY = 1.5


