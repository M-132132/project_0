
import os
import pickle

import re


def get_lstm_dataset(data_dir, data_idx=None):
    input_data = []
    for i in range(2, 61):
        if data_idx is not None:
            if data_idx != i:
                continue
        idx_str = '{0:02}'.format(i)
        pickle_in = open(data_dir + "/time_series_"+idx_str+"_Normal.pickle", "rb")
        temp_data = pickle.load(pickle_in)
        # print("Loaded "+idx_str+" data pack")
        input_data.extend(temp_data)
    return input_data


def get_lstm_NGSIM_dataset(data_dir, ngsim_files):
    input_data = []
    for ngsim_file in ngsim_files:
        time_match = re.search(r'(\d{4})-(\d{4})', ngsim_file)
        if time_match:
            time_str = f"{time_match.group(1)}-{time_match.group(2)}"
        else:
            # 尝试匹配am格式 (如 0750am-0805am)
            time_match = re.search(r'(\d{4}am)-(\d{4}am)', ngsim_file)
            if time_match:
                time_str = f"{time_match.group(1)}-{time_match.group(2)}"
            else:
                # 如果都匹配不到，使用原始文件名（不含扩展名）
                time_str = os.path.splitext(ngsim_file)[0]
        
        filename = f"time_series_{time_str}_Normal.pickle"
        full_path = os.path.join(data_dir, filename)
        
        pickle_in = open(full_path, "rb")
        temp_data = pickle.load(pickle_in)
        input_data.extend(temp_data)
    return input_data


def get_dual_trans_NGSIM_dataset(data_dir, ngsim_files):
    input_data = []
    for ngsim_file in ngsim_files:
        time_match = re.search(r'(\d{4})-(\d{4})', ngsim_file)
        if time_match:
            time_str = f"{time_match.group(1)}-{time_match.group(2)}"
        else:
            # 尝试匹配am格式 (如 0750am-0805am)
            time_match = re.search(r'(\d{4}am)-(\d{4}am)', ngsim_file)
            if time_match:
                time_str = f"{time_match.group(1)}-{time_match.group(2)}"
            else:
                # 如果都匹配不到，使用原始文件名（不含扩展名）
                time_str = os.path.splitext(ngsim_file)[0]
        
        filename = f"dual_trans_series_{time_str}.pickle"
        full_path = os.path.join(data_dir, filename)
        
        pickle_in = open(full_path, "rb")
        temp_data = pickle.load(pickle_in)
        input_data.extend(temp_data)
    return input_data