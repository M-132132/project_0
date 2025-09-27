
import os
import pickle

import torch.utils.data as data
import utils_data.IO as IO


class DataSetTest(data.Dataset):
    def __init__(self, option):
        abs_dir = IO.get_proj_abs_dir()
        root = abs_dir + '/dataset_lc/data_dop/'
        # testing, training
        self.option = option
        self.data_file = os.path.join(root, f"result_{self.option}_CNN_FC.pickle")

        with open(self.data_file, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0]["ego_dop"], \
               self.data[index][0]["sur_dop"], \
               self.data[index][0]["ego_vector"], \
               self.data[index][0]["other_info"], \
               self.data[index][1]


class DataSet(data.Dataset):
    def __init__(self, option):
        abs_dir = IO.get_proj_abs_dir()
        root = abs_dir + '/dataset_lc/data_dop/'
        # testing, training
        self.option = option
        self.data_file = os.path.join(root, f"result_{self.option}_CNN_FC.pickle")

        with open(self.data_file, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0]["ego_dop"], \
               self.data[index][0]["sur_dop"], \
               self.data[index][0]["ego_vector"], \
               self.data[index][1]
