import sys
import os
# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 路径
sys.path.append(project_root)

import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool
import h5py
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from utils_datasets_traj import common_utils
from utils_datasets_traj.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline  # 导入数据集处理工具函数
from utils_datasets_traj.types import object_type, polyline_type  # 导入对象类型和多边形类型定义

from utils.visualization import check_loaded_data  # 导入可视化工具
from utils.path_manager import path_manager

from functools import lru_cache  # 导入LRU缓存装饰器

import hydra
from omegaconf import OmegaConf
from omegaconf import ListConfig

# 设置默认值为0
default_value = 0
# 创建对象类型和多边形类型的字典，并设置默认值
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)


class BaseDataset(Dataset):

    """
    基础数据集类，继承自torch.utils_traj_attr.data.Dataset
    用于处理和加载场景数据
    """
    def __init__(self, config=None, is_validation=False):
        # 初始化数据集路径和配置
        if is_validation:
            self.data_path = config['val_data_path']
        else:
            self.data_path = config['train_data_path']
            
        # 处理路径列表
        if isinstance(self.data_path, ListConfig):
            self.data_path = [path_manager.resolve_path(path) for path in self.data_path]
        else:
            self.data_path = path_manager.resolve_path(self.data_path)

        self.is_validation = is_validation
        self.config = config
        self.data_loaded_memory = []
        self.file_cache = {}
        self.load_data()  # 加载数据

    def load_data(self):
        """
        加载数据集的主要方法
        根据配置加载数据，可以选择是否使用缓存
        """
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')  # 加载验证数据
        else:
            print('Loading training data...')  # 加载训练数据

        # 遍历所有数据路径
        for cnt, data_path in enumerate(self.data_path):
            # 解析数据集名称和阶段
            # phase, dataset_name = data_path.split('/')[-2],data_path.split('/')[-1]
            parts = data_path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                phase, dataset_name = parts[-2], parts[-1]
            else:
                raise ValueError(f"Invalid data_path: {data_path}. Expecting at least 2 levels of directory.")
            
            # 设置缓存路径
            cache_path = path_manager.resolve_path(self.config['cache_path'])
            self.cache_path = os.path.join(cache_path, dataset_name, phase)

            # 获取当前数据集的使用量
            data_usage_this_dataset = self.config['max_data_num'][cnt]
            # 获取起始帧
            self.starting_frame = self.config['starting_frame'][cnt]
            # 如果使用缓存或多进程分布式训练
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                # 如果缓存路径已存在且不覆盖缓存
                if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
                    print('Warning: cache path {} already exists, skip '.format(self.cache_path))
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:

                    # 读取数据集摘要
                    _, summary_list, mapping = read_dataset_summary(data_path)

                    # 如果缓存路径存在则删除
                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    # 创建新的缓存路径
                    os.makedirs(self.cache_path, exist_ok=True)
                    # 设置进程数为CPU核心数的一半
                    process_num = os.cpu_count()//2
                    print('Using {} processes to load data...'.format(process_num))

                    # 将数据分割为多个块
                    data_splits = np.array_split(summary_list, process_num)

                    # 为每个进程准备数据
                    data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
                    # save the data_splits in a tmp directory
                    os.makedirs('tmp', exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp', '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)

                    # results = self.process_data_chunk(0)
                    with Pool(processes=process_num) as pool:
                        results = pool.map(self.process_data_chunk, list(range(process_num)))

                    # concatenate the results
                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)

                    data_list = list(file_list.items())
                    np.random.shuffle(data_list)
                    if not self.is_validation:
                        # randomly sample data_usage number of data
                        file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list), data_path))
            self.data_loaded.update(file_list)

            if self.config['store_data_in_memory']:
                print('Loading data into memory...')
                for data_path in file_list.keys():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                print('Loaded {} data into memory'.format(len(self.data_loaded_memory)))

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')

    def process_data_chunk(self, worker_index):
        """
        处理数据块的方法，从临时文件中加载数据，处理后保存到HDF5文件中
        Args:
            worker_index: 工作进程索引，用于标识不同的数据块
        Returns:
            file_list: 处理后的文件列表，包含文件路径和相关信息
        """
        # 从临时pickle文件中加载数据块
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        # 创建HDF5文件并处理数据
        with h5py.File(hdf5_path, 'w') as f:
            # 遍历数据列表中的每个文件
            for cnt, file_name in enumerate(data_list):
                # 进度显示，仅worker_index为0时打印
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                # 读取场景数据
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    # 三步处理流程：预处理、处理、后处理
                    output = self.preprocess(scenario)  # 预处理

                    output = self.process(output)  # 处理

                    output = self.postprocess(output)  # 后处理

                except Exception as e:
                    # 处理异常情况
                    print('Warning: {} in {}'.format(e, file_name))
                    output = None

                # 如果处理结果为None则跳过
                if output is None: continue

                # 将处理结果保存到HDF5文件中
                for i, record in enumerate(output):
                    # 创建组名
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    # 将记录中的每个键值对保存为数据集
                    for key, value in record.items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)
                    # 保存文件信息
                    file_info = {}
                    kalman_difficulty = np.stack([x['kalman_difficulty'] for x in output])
                    file_info['kalman_difficulty'] = kalman_difficulty
                    file_info['h5_path'] = hdf5_path
                    file_list[grp_name] = file_info
                # 释放内存
                del scenario
                del output

        return file_list

    def preprocess(self, scenario):
        """
        预处理方法，对原始场景数据进行初步处理和转换
        Args:
            scenario: 场景数据，包含交通灯状态、轨迹数据和地图特征等信息
        Returns:
            处理后的场景数据，包含轨迹信息、动态地图信息和地图信息等
        """
        # 提取场景数据中的关键信息
        traffic_lights = scenario['dynamic_map_states']  # 交通灯状态
        tracks = scenario['tracks']  # 轨迹数据
        map_feat = scenario['map_features']  # 地图特征

        # 获取配置参数
        past_length = self.config['past_len']  # 过去时间步长
        future_length = self.config['future_len']  # 未来时间步长
        total_steps = past_length + future_length  # 总时间步长
        starting_fame = self.starting_frame  # 起始帧
        ending_fame = starting_fame + total_steps  # 结束帧
        trajectory_sample_interval = self.config['trajectory_sample_interval']  # 轨迹采样间隔
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)  # 生成频率掩码

        # 初始化轨迹信息字典
        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }

        # 处理每条轨迹数据
        for k, v in tracks.items():
            state = v['state']
            # 确保状态数据的维度正确
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            # 合并所有状态信息
            all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                         state['velocity'], state['valid']]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                all_state = np.pad(all_state, ((ending_fame - all_state.shape[0], 0), (0, 0)))
            all_state = all_state[starting_fame:ending_fame]

            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'

            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]

        # x,y,z,type
        map_infos = {
            'lane': [],  # 车道信息
            'road_line': [],  # 道路线信息
            'road_edge': [],  # 道路边缘信息
            'stop_sign': [],  # 停止标志信息
            'crosswalk': [],  # 人行横道信息
            'speed_bump': [],  # 减速带信息
        }
        polylines = []  # 存储所有多边形线段
        point_cnt = 0  # 点计数器
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]  # 获取多边形类型
            if polyline_type_ == 0:
                continue

            cur_info = {'id': k}  # 当前特征信息
            cur_info['type'] = v['type']  # 特征类型
            if polyline_type_ in [1, 2, 3]:  # 车道相关
                cur_info['speed_limit_mph'] = v.get('speed_limit_mph', None)  # 速度限制
                cur_info['interpolating'] = v.get('interpolating', None)  # 插值信息
                cur_info['entry_lanes'] = v.get('entry_lanes', None)  # 入口车道
                try:
                    # 左边界信息
                    cur_info['left_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # 道路边界类型
                    } for x in v['left_neighbor']
                    ]
                    # 右边界信息
                    cur_info['right_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # 道路边界类型
                    } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []  # 左边界为空
                    cur_info['right_boundary'] = []  # 右边界为空
                polyline = v['polyline']  # 获取车道多边形线段
                polyline = interpolate_polyline(polyline)  # 插值处理
                map_infos['lane'].append(cur_info)
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:  # 道路线相关
                try:
                    polyline = v['polyline']  # 获取多边形线段
                except:
                    polyline = v['polygon']  # 获取多边形
                polyline = interpolate_polyline(polyline)  # 插值处理
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [15, 16]:  # 道路边缘相关
                polyline = v['polyline']  # 获取多边形线段
                polyline = interpolate_polyline(polyline)  # 插值处理
                cur_info['type'] = 7  # 设置类型
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [17]:  # 停止标志相关
                cur_info['lane_ids'] = v['lane']  # 车道ID
                cur_info['position'] = v['position']  # 位置
                map_infos['stop_sign'].append(cur_info)  # 添加到停止标志信息
                polyline = v['position'][np.newaxis]  # 获取位置多边形
            elif polyline_type_ in [18]:  # 人行横道相关
                map_infos['crosswalk'].append(cur_info)  # 添加到人行横道信息
                polyline = v['polygon']  # 获取多边形
            elif polyline_type_ in [19]:  # 减速带相关
                map_infos['crosswalk'].append(cur_info)  # 添加到人行横道信息
                polyline = v['polygon']  # 获取多边形
            if polyline.shape[-1] == 2:  # 如果是二维点，转换为三维点
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = get_polyline_dir(polyline)  # 获取多边形方向
                type_array = np.zeros([polyline.shape[0], 1])  # 创建类型数组
                type_array[:] = polyline_type_  # 设置类型
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)  # 合并信息
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)  # 创建空多边形
            polylines.append(cur_polyline)  # 添加到多边形列表
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))  # 设置多边形索引
            point_cnt += len(cur_polyline)  # 更新点计数

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)  # 合并所有多边形
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)  # 创建空多边形数组
        map_infos['all_polylines'] = polylines  # 存储所有多边形

        dynamic_map_infos = {
            'lane_id': [],  # 车道ID
            'state': [],  # 状态
            'stop_point': []  # 停止点
        }
        for k, v in traffic_lights.items():  # 遍历交通灯数据
            lane_id, state, stop_point = [], [], []  # 初始化列表
            for cur_signal in v['state']['object_state']:  # 遍历信号状态
                lane_id.append(str(v['lane']))  # 添加车道ID
                state.append(cur_signal)  # 添加信号状态
                if type(v['stop_point']) == list:  # 处理停止点
                    stop_point.append(v['stop_point'])
                else:
                    stop_point.append(v['stop_point'].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]  # 截取到总时间步长
            state = state[:total_steps]  # 截取到总时间步长
            stop_point = stop_point[:total_steps]  # 截取到总时间步长
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))  # 添加车道ID
            dynamic_map_infos['state'].append(np.array([state]))  # 添加状态
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))  # 添加停止点

        ret = {
            'track_infos': track_infos,  # 轨迹信息
            'dynamic_map_infos': dynamic_map_infos,  # 动态地图信息
            'map_infos': map_infos  # 地图信息
        }
        ret.update(scenario['metadata'])  # 更新元数据
        ret['timestamps_seconds'] = ret.pop('ts')  # 时间戳
        ret['current_time_index'] = self.config['past_len'] - 1  # 当前时间索引
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])  # 自主导航车辆轨迹索引

        if self.config['only_train_on_ego']:  # 如果只在自车训练
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],  # 轨迹索引
                'difficulty': [0],  # 难度
                'object_type': [MetaDriveType.VEHICLE]  # 对象类型
            }
        elif ret.get('tracks_to_predict', None) is None:  # 如果没有指定预测轨迹
            filtered_tracks = self.trajectory_filter(ret)  # 过滤轨迹
            sample_list = list(filtered_tracks.keys())  # 获取采样列表
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],  # 轨迹索引
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],  # 对象类型
            }
        else:  # 如果有指定预测轨迹
            sample_list = list(ret['tracks_to_predict'].keys())  # 获取采样列表
            sample_list = list(set(sample_list))  # 去重
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],  # 轨迹索引
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],  # 对象类型
            }

        ret['tracks_to_predict'] = tracks_to_predict  # 设置预测轨迹

        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]  # 地图中心

        ret['track_length'] = total_steps  # 轨迹长度
        return ret  # 返回处理后的结果

    def process(self, internal_format):
            """
            处理内部格式数据，提取和转换轨迹信息，生成预测所需的特征数据
            
            参数:
                internal_format: 包含场景信息的字典，包含轨迹、地图、时间戳等数据
                
            返回:
                处理后的数据列表，每个元素是一个包含预测所需特征的字典
            """
            # 获取场景ID
            info = internal_format
            scene_id = info['scenario_id']
    
            # 获取自车轨迹索引和当前时间索引
            sdc_track_index = info['sdc_track_index']
            current_time_index = info['current_time_index']
            # 将时间戳转换为numpy数组，类型为float32
            timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)
    
            # 获取所有轨迹信息
            track_infos = info['track_infos']
    
            # 获取需要预测的轨迹索引和对象类型
            track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
            obj_types = np.array(track_infos['object_type'])
            # 获取完整轨迹数据，形状为(对象数量, 时间戳数量, 10)
            obj_trajs_full = track_infos['trajs']
            # 分割历史轨迹和未来轨迹
            obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]  # 历史轨迹
            obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]  # 未来轨迹
    
            # 获取感兴趣的代理对象
            center_objects, track_index_to_predict = self.get_interested_agents(
                track_index_to_predict=track_index_to_predict,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types, scene_id=scene_id
            )
            # 如果没有感兴趣的代理对象，直接返回None
            if center_objects is None: return None
    
            # 获取样本数量
            sample_num = center_objects.shape[0]
    
            # 获取代理对象的数据，包括轨迹、掩码、位置等信息
            (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
             obj_trajs_future_mask, center_gt_trajs,
             center_gt_trajs_mask, center_gt_final_valid_idx,
             track_index_to_predict_new) = self.get_agent_data(
                center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
                track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
                timestamps=timestamps, obj_types=obj_types
            )
    
            # 构建返回字典，包含各种特征数据
            ret_dict = {
                'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
                'obj_trajs': obj_trajs_data,
                'obj_trajs_mask': obj_trajs_mask,
                'track_index_to_predict': track_index_to_predict_new,  # 用于选择中心特征
                'obj_trajs_pos': obj_trajs_pos,
                'obj_trajs_last_pos': obj_trajs_last_pos,
    
                'center_objects_world': center_objects,
                'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
                'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
                'map_center': info['map_center'],
    
                'obj_trajs_future_state': obj_trajs_future_state,
                'obj_trajs_future_mask': obj_trajs_future_mask,
                'center_gt_trajs': center_gt_trajs,
                'center_gt_trajs_mask': center_gt_trajs_mask,
                'center_gt_final_valid_idx': center_gt_final_valid_idx,
                'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
            }
    
            # 检查地图数据是否为空，如果为空则设置默认值并打印警告
            if info['map_infos']['all_polylines'].__len__() == 0:
                info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
                print(f'Warning: empty HDMap {scene_id}')
    
            # 根据配置选择手动分割或自动分割地图数据
            if self.config.manually_split_lane:
                map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
                    center_objects=center_objects, map_infos=info['map_infos'])
            else:
                map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
                    center_objects=center_objects, map_infos=info['map_infos'])
    
            # 将地图数据添加到返回字典中
            ret_dict['map_polylines'] = map_polylines_data
            ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
            ret_dict['map_polylines_center'] = map_polylines_center
    
            # 根据配置掩码未使用的属性
            masked_attributes = self.config['masked_attributes']
            if 'z_axis' in masked_attributes:
                ret_dict['obj_trajs'][..., 2] = 0
                ret_dict['map_polylines'][..., 2] = 0
            if 'size' in masked_attributes:
                ret_dict['obj_trajs'][..., 3:6] = 0
            if 'velocity' in masked_attributes:
                ret_dict['obj_trajs'][..., 25:27] = 0
            if 'acceleration' in masked_attributes:
                ret_dict['obj_trajs'][..., 27:29] = 0
            if 'heading' in masked_attributes:
                ret_dict['obj_trajs'][..., 23:25] = 0
    
            # 将所有数据转换为float32类型
            for k, v in ret_dict.items():
                if isinstance(v, np.ndarray) and v.dtype == np.float64:
                    ret_dict[k] = v.astype(np.float32)
    
            # 复制地图中心点以匹配样本数量
            ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
            ret_dict['dataset_name'] = [info['dataset']] * sample_num
    
            # 构建最终返回列表，每个元素是一个包含单个样本特征的字典
            ret_list = []
            for i in range(sample_num):
                ret_dict_i = {}
                for k, v in ret_dict.items():
                    ret_dict_i[k] = v[i]
                ret_list.append(ret_dict_i)
    
            return ret_list

    def postprocess(self, output):

        # Add the trajectory difficulty
        get_kalman_difficulty(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type(output)

        return output

    def collate_fn(self, data_list):
        batch_list = []
        for batch in data_list:
            batch_list.append(batch)

        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            # if val_list is str:
            try:
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except:
                input_dict[key] = val_list

        input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()

        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        return batch_dict

    def __len__(self):
        return len(self.data_loaded_keys)

    @lru_cache(maxsize=None)
    def _get_file(self, file_path):
        return h5py.File(file_path, 'r')

    def __getitem__(self, idx):
        file_key = self.data_loaded_keys[idx]
        file_info = self.data_loaded[file_key]
        file_path = file_info['h5_path']

        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)

        group = self.file_cache[file_path][file_key]
        record = {k: group[k][()].decode('utf-8') if group[k].dtype.type == np.bytes_ else group[k][()] for k in group.keys()}

        return record

    def close_files(self):
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()

    def get_data_list(self, data_usage):
        file_list_path = os.path.join(self.cache_path, 'file_list.pkl')
        if os.path.exists(file_list_path):
            data_loaded = pickle.load(open(file_list_path, 'rb'))
        else:
            raise ValueError('Error: file_list.pkl not found')

        data_list = list(data_loaded.items())
        np.random.shuffle(data_list)

        if not self.is_validation:
            # randomly sample data_usage number of data
            data_loaded = dict(data_list[:data_usage])
        else:
            data_loaded = dict(data_list)
        return data_loaded

    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types
    ):
        """
        获取智能体数据的函数，处理过去和未来的轨迹信息，并返回多种编码后的数据
        
        参数:
            center_objects: 中心对象数据
            obj_trajs_past: 过去轨迹数据
            obj_trajs_future: 未来轨迹数据
            track_index_to_predict: 需要预测的轨迹索引
            sdc_track_index: 自主导驶车辆(SDC)的轨迹索引
            timestamps: 时间戳
            obj_types: 对象类型
            
        返回:
            包含处理后的轨迹数据和掩码的元组
        """
        num_center_objects = center_objects.shape[0]  # 获取中心对象的数量
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape  # 获取对象数量、时间戳维度和框维度
        # 将轨迹转换为中心坐标
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        # 创建对象one-hot掩码，用于表示不同类型的对象
        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1  # 类型1的掩码
        object_onehot_mask[:, obj_types == 2, :, 1] = 1  # 类型2的掩码
        object_onehot_mask[:, obj_types == 3, :, 2] = 1  # 类型3的掩码
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1  # 预测对象的掩码
        object_onehot_mask[:, sdc_track_index, :, 4] = 1  # SDC Self-Driving Car 的掩码

        # 创建时间嵌入向量
        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1  # 时间位置编码
        object_time_embedding[:, :, :, -1] = timestamps  # 添加时间戳信息

        # 创建方向嵌入向量
        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])  # 方向正弦值
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])  # 方向余弦值

        # 计算加速度
        vel = obj_trajs[:, :, :, 7:9]  # 速度
        vel_pre = np.roll(vel, shift=1, axis=2)  # 前一时刻速度
        acce = (vel - vel_pre) / 0.1  # 加速度计算
        acce[:, :, 0, :] = acce[:, :, 1, :]  # 处理初始时刻加速度

        # 拼接所有特征数据
        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],  # 基本轨迹信息
            object_onehot_mask,  # 对象类型掩码
            object_time_embedding,  # 时间嵌入
            object_heading_embedding,  # 方向嵌入
            obj_trajs[:, :, :, 7:9],  # 速度信息
            acce,  # 加速度信息
        ], axis=-1)

        # 处理无效轨迹数据
        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        # 处理未来轨迹数据
        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        # 获取中心对象的真实轨迹
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # 验证过去轨迹数据的有效性
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        # 根据有效掩码筛选数据
        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        # 获取轨迹位置信息
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        # 获取中心对象的真实有效索引
        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        # 根据距离选择最近的智能体
        max_num_agents = self.config['max_num_agents']
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10  # 将无效距离设为极大值
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]  # 选择最近的智能体

        # 扩展维度以便后续处理
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        # 使用选择的索引筛选数据
        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        # 填充数据以达到固定数量
        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        # 返回处理后的所有数据
        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):

        """
        根据预测目标索引、轨迹信息、当前时间步、目标类型和场景ID获取感兴趣的智能体
        
        参数:
            track_index_to_predict: list, 需要预测的目标索引列表
            obj_trajs_full: numpy.ndarray, 所有对象的完整轨迹数据
            current_time_index: int, 当前时间步的索引
            obj_types: list, 对象类型列表
            scene_id: int, 场景ID
            
        返回:
            tuple: (center_objects, track_index_to_predict)
                center_objects: numpy.ndarray, 感兴趣的中心对象轨迹信息
                track_index_to_predict: numpy.ndarray, 被选中的目标索引
        """
        center_objects_list = []  # 用于存储中心对象轨迹的列表
        track_index_to_predict_selected = []  # 用于存储被选中的目标索引列表
        # 从配置中获取选定的对象类型
        selected_type = self.config['object_type']
        selected_type = [object_type[x] for x in selected_type]  # 将类型名称转换为对应的数值
        # 遍历所有需要预测的目标索引
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]  # 获取当前对象索引

            # 检查对象在当前时间步是否有效
            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
                continue  # 如果对象无效，跳过该对象
            # 检查对象类型是否在选定的类型中
            if obj_types[obj_idx] not in selected_type:
                continue  # 如果类型不匹配，跳过该对象

            # 将有效的对象轨迹和索引添加到列表中
            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        # 如果没有找到有效的中心对象，打印警告并返回
        if len(center_objects_list) == 0:
            print(f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
            return None, []
        # 将中心对象列表转换为numpy数组
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(self, obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None):
        """
        将轨迹转换为中心坐标系
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        # 获取轨迹数据的维度信息
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        # 获取中心对象的数量
        num_center_objects = center_xyz.shape[0]
        # 断言检查：确保中心坐标和中心朝向的数量一致
        assert center_xyz.shape[0] == center_heading.shape[0]
        # 断言检查：确保中心坐标是二维或三维的
        assert center_xyz.shape[1] in [3, 2]

        # 将轨迹数据复制以匹配中心对象的数量
        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        # 减去中心坐标，实现平移变换
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        # 绕Z轴旋转轨迹点，调整坐标系方向
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        # 调整朝向角度，减去中心朝向
        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def get_map_data(self, center_objects, map_infos):
        """
        获取地图数据并进行处理，将地图信息转换为以中心对象为参考的坐标系
        参数:
            center_objects: 中心对象数组，包含位置和旋转信息
            map_infos: 包含所有地图信息的字典
        返回:
            map_polylines: 处理后的地图线段数据
            map_polylines_mask: 地图线段掩码
            map_polylines_center: 地图线段中心点
        """
        # 获取中心对象的数量
        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            """
            将邻近线段转换到以中心对象为参考的坐标系中
            参数:
                neighboring_polylines: 需要转换的线段数据
            返回:
                转换后的线段数据
            """
            # 平移线段，使其相对于中心对象的位置
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            # 旋转线段，使其与中心对象的朝向对齐
            neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        # 复制并扩展地图线段数据以匹配中心对象的数量
        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

        # 将线段转换到中心对象坐标系
        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        # 获取源线段的最大数量
        num_of_src_polylines = self.config['max_num_roads']
        # 保存转换后的线段数据
        map_infos['polyline_transformed'] = map_polylines

        # 获取所有转换后的线段
        all_polylines = map_infos['polyline_transformed']
        # 获取每条车道最大点数
        max_points_per_lane = self.config.get('max_points_per_lane', 20)
        # 获取线段类型
        line_type = self.config.get('line_type', [])
        # 获取地图范围
        map_range = self.config.get('map_range', None)
        # 获取地图中心偏移量
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        # 获取智能体数量
        num_agents = all_polylines.shape[0]
        # 初始化线段列表和掩码列表
        polyline_list = []
        polyline_mask_list = []

        # 遍历地图信息，处理不同类型的线段
        for k, v in map_infos.items():
            # 跳过不需要处理的线段
            if k == 'all_polylines' or k not in line_type:
                continue
            # 跳过空线段
            if len(v) == 0:
                continue
            # 处理每个线段字典
            for polyline_dict in v:
                # 获取线段索引
                polyline_index = polyline_dict.get('polyline_index', None)
                # 获取线段数据
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                # 应用中心偏移
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                # 创建范围掩码
                in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

                # 初始化线段索引列表
                segment_index_list = []
                # 为每个智能体找到有效线段
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                # 找到最大线段数
                max_segments = max([len(x) for x in segment_index_list])

                # 初始化线段列表和掩码列表
                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

                # 处理每个智能体的线段
                for i in range(polyline_segment.shape[0]):
                    # 如果线段不在范围内，跳过
                    if in_range_mask[i].sum() == 0:
                        continue
                    # 获取当前线段
                    segment_i = polyline_segment[i]
                    # 获取当前线段的索引
                    segment_index = segment_index_list[i]
                    # 处理每个线段
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        # 如果线段点数超过最大值，进行下采样
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                            segment_mask_list[i, num] = 1
                        else:
                            # 否则，直接填充
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1

                # 将处理后的线段添加到列表中
                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        
        # 如果没有有效的线段，返回零数组
        if len(polyline_list) == 0:
            return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
                (num_agents, 0, max_points_per_lane))
        
        # 合并所有线段
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        # 计算线段中心距离
        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        # 获取最近的线段索引
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # 确保索引形状正确
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        # 选择最近的线段
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

        # 填充线段到固定数量, max_num_roads决定,
        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        # 计算线段中心点
        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                                  a_max=None)  # (num_center_objects, num_polylines, 3)

        # 处理线段位置
        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        # 获取线段类型并进行one-hot编码
        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        # 合并线段数据和类型
        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        # 对无效线段进行掩码处理
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center


    def get_manually_split_map_data(self, center_objects, map_infos):
        """
        获取手动分割的地图数据，将地图信息转换为以中心对象为参考的坐标系
        
        Args:
            center_objects (num_center_objects, 10): 中心对象信息数组，包含 [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict): 地图信息字典，包含以下键:
                all_polylines (num_points, 7): 所有多边形线段信息，包含 [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2): 地图中心偏移量 [offset_x, offset_y]
            
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9):
                转换后的地图多边形线段数据，包含 [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline):
                地图多边形线段的有效掩码
            map_polylines_center (num_center_objects, num_topk_polylines, 3):
                每个多边形线段的中心点坐标
        """
        num_center_objects = center_objects.shape[0]
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))

        # 将坐标转换为中心对象坐标系
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            # 平移坐标，使中心对象位于原点
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            # 绕Z轴旋转坐标，使中心对象的朝向为X轴正方向
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)
            # 旋转方向向量
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)

            # 使用前一个点进行映射
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:3]
            xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = np.concatenate((neighboring_polylines, xy_pos_pre), axis=-1)

            # 应用有效掩码
            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = map_infos['all_polylines'].copy()
        center_objects = center_objects

        point_dim = polylines.shape[-1]

        # 从配置中获取参数
        point_sampled_interval = self.config['point_sampled_interval']
        vector_break_dist_thresh = self.config['vector_break_dist_thresh']
        num_points_each_polyline = self.config['num_points_each_polyline']

        # 对多边形线段进行采样
        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                       axis=-1)  # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        # 检测线段断裂点
        break_idxs = \
            (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4],
                            axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        # 将单个多边形线段添加到结果列表
        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        # 处理每个多边形线段
        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        # 堆叠所有多边形线段
        batch_polylines = np.stack(ret_polylines, axis=0)
        batch_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        # 为每个中心对象收集最近的多边形线段
        num_of_src_polylines = self.config['max_num_roads']

        if len(batch_polylines) > num_of_src_polylines:
            # 计算多边形线段的中心点
            polyline_center = np.sum(batch_polylines[:, :, 0:2], axis=1) / np.clip(
                np.sum(batch_polylines_mask, axis=1)[:, None].astype(float), a_min=1.0, a_max=None)
            # 旋转中心偏移量
            center_offset_rot = np.tile(np.array(center_offset, dtype=np.float32)[None, :], (num_center_objects, 1))

            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot[:, None, :],
                angle=center_objects[:, 6]
            )

            # 计算地图中心位置
            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot[:, 0]

            # 计算距离并选择最近的多边形线段
            dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :], axis=-1)

            # 获取top-k最近的多边形线段
            topk_idxs = np.argsort(dist, axis=1)[:, :num_of_src_polylines]
            map_polylines = batch_polylines[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)

        else:
            # 如果多边形线段数量不足，进行填充
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 0)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 0)

            map_polylines = np.pad(map_polylines,
                                   ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
            map_polylines_mask = np.pad(map_polylines_mask,
                                        ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        # 转换为中心坐标系
        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        # 计算多边形线段的中心点
        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(np.float32)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1)[:, :, np.newaxis].astype(float),
                                                  a_min=1.0, a_max=None)

        # 提取地图类型和前一位置信息
        map_types = map_polylines[:, :, :, 6]
        xy_pos_pre = map_polylines[:, :, :, 7:]
        map_polylines = map_polylines[:, :, :, :6]
        # 对地图类型进行one-hot编码，共14种类型，使用20作为保留类型
        map_types = np.eye(20)[map_types.astype(int)]

        # 拼接所有特征
        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)

        return map_polylines, map_polylines_mask, map_polylines_center
    
    @staticmethod
    def sample_from_distribution(original_array, m=100):
        # 定义分布区间和对应的百分比
        distribution = [
            ("-10,0", 0),
            ("0,10", 23.952629169758517),
            ("10,20", 24.611144221251667),
            ("20,30.0", 21.142773679220554),
            ("30,40.0", 15.996653629820514),
            ("40,50.0", 9.446714336574939),
            ("50,60.0", 3.7812939732733786),
            ("60,70", 0.8821063091988663),
            ("70,80.0", 0.1533644322320915),
            ("80,90.0", 0.027777741552241064),
            ("90,100.0", 0.005542507117231198),
        ]
    
        # 定义区间边界并计算每个区间的采样数量
        bins = np.array([float(range_.split(',')[1]) for range_, _ in distribution])
        sample_sizes = np.array([round(perc / 100 * m) for _, perc in distribution])
    
        # 将原始数组分配到各个区间
        bin_indices = np.digitize(original_array, bins)
    
        # 从每个区间中进行采样
        sampled_indices = []
        for i, size in enumerate(sample_sizes):
            # 找出落在当前区间内的原始数组索引
            indices_in_bin = np.where(bin_indices == i)[0]
            # 无重复采样以避免重复
            sampled_indices_in_bin = np.random.choice(indices_in_bin, size=min(size, len(indices_in_bin)),
                                                      replace=False)
            sampled_indices.extend(sampled_indices_in_bin)
    
        # 提取采样元素及其原始索引
        sampled_array = original_array[sampled_indices]
        print('总采样数:', len(sampled_indices))
        # 验证采样分布（可选，用于演示）
        for i, (range_, _) in enumerate(distribution):
            print(
                f"区间 {range_}: 预期 {distribution[i][1]}%, 实际 "
                f"{len(np.where(bin_indices[sampled_indices] == i)[0]) / len(sampled_indices) * 100}%")
    
        return sampled_array, sampled_indices

    @staticmethod
    def trajectory_filter(data):
        # 提取轨迹数据、当前时间索引和对象摘要信息
        trajs = data['track_infos']['trajs']
        current_idx = data['current_time_index']
        obj_summary = data['object_summary']

        # 初始化需要预测的轨迹字典
        tracks_to_preidct = {}
        
        # 遍历所有对象
        for idx,(k,v) in enumerate(obj_summary.items()):
            type = v['type']  # 获取对象类型
            positions = trajs[idx, :, 0:2]  # 获取对象的位置信息 (x,y)
            validity = trajs[idx, :, -1]  # 获取轨迹的有效性标记
            
            # 过滤条件1：只保留车辆、行人和自行车
            if type not in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']: continue
            
            # 过滤条件2：有效轨迹长度比例必须大于50%
            valid_ratio = v['valid_length']/v['track_length']
            if valid_ratio < 0.5: continue
            
            # 过滤条件3：车辆的移动距离必须大于2.0米
            moving_distance = v['moving_distance']
            if moving_distance < 2.0 and type=='VEHICLE': continue
            
            # 过滤条件4：在当前时间点必须是有效的
            is_valid_at_m = validity[current_idx]>0
            if not is_valid_at_m: continue

            # 获取未来轨迹的有效性掩码
            future_mask = validity[current_idx+1:]
            future_mask[-1]=0  # 将最后一个时间点设为无效
            
            # 找到第一个无效时间点的索引
            idx_of_first_zero = np.where(future_mask == 0)[0]
            idx_of_first_zero = len(future_mask) if len(idx_of_first_zero) == 0 else idx_of_first_zero[0]

            # 以下是被注释掉的卡尔曼滤波相关代码
            # past_traj = positions[:current_idx+1, :]  # 过去轨迹 Time X (x,y)
            # gt_future = positions[current_idx+1:, :]  # 未来轨迹
            # valid_past = count_valid_steps_past(validity[:current_idx+1])  # 计算过去有效步数
            # past_trajectory_valid = past_traj[-valid_past:, :]  # 有效的过去轨迹
            # try:
            #     kalman_traj = estimate_kalman_filter(past_trajectory_valid, idx_of_first_zero)  # 卡尔曼滤波预测
            #     kalman_diff = calculate_epe(kalman_traj, gt_future[idx_of_first_zero-1])  # 计算预测误差
            # except:
            #     continue
            # if kalman_diff < 20: continue  # 如果预测误差太小，跳过

            # 将符合条件的轨迹添加到预测列表中
            tracks_to_preidct[k] = {
                'track_index': idx,  # 轨迹索引
                'track_id': k,  # 轨迹ID
                'difficulty': 0,  # 难度等级（初始设为0）
                'object_type': type  # 对象类型
            }

        return tracks_to_preidct


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def draw_figures(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,
                                               collate_fn=train_set.collate_fn)
    # for data in train_loader:
    #     inp = data['input_dict']
    #     plt = check_loaded_data(inp, 0)
    #     plt.show()

    concat_list = [4, 4, 4, 4, 4, 4, 4, 4]
    images = []
    for n, data in tqdm(enumerate(train_loader)):
        for i in range(data['batch_size']):
            plt = check_loaded_data(data['input_dict'], i)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)
        if len(images) >= sum(concat_list):
            break
    final_image = concatenate_varying(images, concat_list)
    final_image.show()

    # kalman_dict = {}
    # # create 10 buckets with length 10 as the key
    # for i in range(10):
    #     kalman_dict[i] = {}
    #
    # data_list = []
    # for data in train_loader:
    #     inp = data['input_dict']
    #     kalman_diff = inp['kalman_difficulty']
    #     for idx,k in enumerate(kalman_diff):
    #         k6 = np.floor(k[2]/10)
    #         if k6 in kalman_dict and len(kalman_dict[k6]) == 0:
    #             kalman_dict[k6]['kalman'] = k[2]
    #             kalman_dict[k6]['data'] = inp
    #             check_loaded_data()
    #


# @hydra.main(version_base=None, config_path="../configs", config_name="config")
@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="config")
def split_data(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    copy_dir = ''
    for data in tqdm(train_set.data_loaded_keys):
        shutil.copy(data, copy_dir)


if __name__ == '__main__':
    from utils_datasets_traj import build_dataset
    from utils.utils_train_traj import set_seed
    import io
    from PIL import Image
    from utils.visualization import concatenate_varying

    split_data()
    # draw_figures()
