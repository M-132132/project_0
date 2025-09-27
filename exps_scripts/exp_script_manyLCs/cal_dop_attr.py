
"""
在 DOP LC 换道模型, 利用不同的归因法计算输入归因值

其中 results 文件夹的截图是利用 lane_change/highd_vis 代码库实现的

"""

from torch.utils.data import DataLoader

import utils_data.utils_save
from dataset_lc.dataset import DataSetTest

from utils_config.traffic_attr.config_attr_doplc import Config
import utils_attr.manyLCs_attr.class_cal_attr_dop as cal_attr_lc


def main():
    opt = Config()
    
    # 初始化模型以及归因计算
    attr_cal_handler = cal_attr_lc.DOPAttrCal(opt)
    # get the dataloader
    data_idx = opt.data_idx
    batch_size = opt.batch_size
    test_loader = DataLoader(DataSetTest("data{}_exp".format(data_idx)),
                             shuffle=False, num_workers=1, batch_size=batch_size)
    
    metadata = {}
    for batch_idx, (ego_dop_data, sur_dop_data, ego_vector_data, other_info, labels) in enumerate(test_loader):
        # 设定限制条件, 部分 ID 的车就不计算了
        if int(other_info[0]) not in opt.car_ids:
            continue
        
        ego_dop_data = ego_dop_data.to(opt.device)
        ego_dop_data.requires_grad = True
        sur_dop_data = sur_dop_data.to(opt.device)
        sur_dop_data.requires_grad = True
        ego_vector_data = ego_vector_data.to(opt.device)
        ego_vector_data.requires_grad = True
        
        input_t = (ego_dop_data, sur_dop_data, ego_vector_data)
        label = utils_data.utils_save.from_tensor_to_np(labels)
        
        metadata['other_info'] = (int(other_info[0]), int(other_info[1]), int(other_info[2]))
        metadata['attr_methods'] = opt.attr_methods
        metadata['label'] = str(label)
        
        attr_res = attr_cal_handler.attr_cal(metadata, input_t)
        aaa = 1
    print()


if __name__ == "__main__":
    main()
