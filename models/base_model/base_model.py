import json
import numpy as np
import torch
import torch.nn as nn
import utils_datasets_traj.common_utils as common_utils
import utils.visualization as visualization
from utils.path_manager import path_manager


class BaseModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_dicts = []

        if config.get('eval_nuscenes', False):
            self.init_nuscenes()
    
    def init_nuscenes(self):
    # 初始化NuScenes数据集和预测配置
    # 当配置中设置了eval_nuscenes为True时，此方法会被调用

        if self.config.get('eval_nuscenes', False):
        # 导入必要的NuScenes库
            from nuscenes import NuScenes
            from nuscenes.eval.prediction.config import PredictionConfig
            from nuscenes.prediction import PredictHelper
        # 解析数据集根目录路径
            dataroot = path_manager.resolve_path(self.config['nuscenes_dataroot'])
        # 初始化NuScenes数据集对象，使用v1.0-trainval版本
            nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot)
            
            # Prediction helper and configs:
            self.helper = PredictHelper(nusc)
            
            with open('models/base_model/nuscenes_config.json', 'r') as f:
                pred_config = json.load(f)
            self.pred_config5 = PredictionConfig.deserialize(pred_config, self.helper)
    
    def forward(self, batch):
        """
        Forward pass for the model
        :param batch: input batch
        :return: prediction: {
                'predicted_probability': (batch_size,modes)),
                'predicted_trajectory': (batch_size,modes, future_len, 2)
                }
                loss (with gradient)
        """
        raise NotImplementedError
    
    def compute_official_evaluation(self, batch_dict, prediction):
        """
        计算官方评估结果的方法，支持Waymo、nuScenes和Argoverse2三种数据集的评估
        参数:
            batch_dict: 包含批次数据的字典
            prediction: 包含预测结果的字典
        返回:
            无返回值，结果会存储在self.pred_dicts中
        """
        if self.config.get('eval_waymo', False):
        # Waymo数据集评估处理
            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']  # 预测得分
            pred_trajs = prediction['predicted_trajectory']    # 预测轨迹
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            
        # 对预测轨迹进行Z轴旋转，并调整到世界坐标系
            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
        # 调整轨迹位置，添加中心物体位置和地图中心偏移
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]
            
            pred_dict_list = []  # 存储预测结果的列表
            
        # 遍历批次中的每个样本
            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],  # 场景ID
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),  # 预测轨迹
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),  # 预测得分
                    'object_id': input_dict['center_objects_id'][bs_idx],  # 物体ID
                    'object_type': input_dict['center_objects_type'][bs_idx],  # 物体类型
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),  # 真实轨迹
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()  # 预测的轨迹索引
                }
                pred_dict_list.append(single_pred_dict)
            
            assert len(pred_dict_list) == batch_dict['batch_size']  # 确保预测数量与批次大小一致
            self.pred_dicts += pred_dict_list  # 将预测结果添加到总预测字典中
        
        elif self.config.get('eval_nuscenes', False):
        # nuScenes数据集评估处理
            from nuscenes.eval.prediction.data_classes import Prediction
            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']  # 预测得分
            pred_trajs = prediction['predicted_trajectory']    # 预测轨迹
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            
        # 对预测轨迹进行Z轴旋转，并调整到世界坐标系
            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
        # 调整轨迹位置，添加中心物体位置和地图中心偏移
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]
            pred_dict_list = []  # 存储预测结果的列表
            
        # 遍历批次中的每个样本
            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'instance': input_dict['scenario_id'][bs_idx].split('_')[1],  # 实例ID
                    'sample': input_dict['scenario_id'][bs_idx].split('_')[2],  # 样本ID
                    'prediction': pred_trajs_world[bs_idx, :, 4::5, 0:2].cpu().numpy(),  # 预测轨迹（每隔4个时间步取一个）
                    'probabilities': pred_scores[bs_idx, :].cpu().numpy(),  # 预测概率
                }
                
            # 将预测结果序列化并添加到列表中
                pred_dict_list.append(
                    Prediction(instance=single_pred_dict["instance"], sample=single_pred_dict["sample"],
                               prediction=single_pred_dict["prediction"],
                               probabilities=single_pred_dict["probabilities"]).serialize())
            
            self.pred_dicts += pred_dict_list  # 将预测结果添加到总预测字典中
        
        elif self.config.get('eval_argoverse2', False):
        # Argoverse2数据集评估处理
            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']  # 预测得分
            pred_trajs = prediction['predicted_trajectory']    # 预测轨迹
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            
        # 对预测轨迹进行Z轴旋转，并调整到世界坐标系
            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
        # 调整轨迹位置，添加中心物体位置和地图中心偏移
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]
            
            pred_dict_list = []  # 存储预测结果的列表
            
        # 遍历批次中的每个样本
            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],  # 场景ID
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),  # 预测轨迹
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),  # 预测得分
                    'object_id': input_dict['center_objects_id'][bs_idx],  # 物体ID
                    'object_type': input_dict['center_objects_type'][bs_idx],  # 物体类型
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),  # 真实轨迹
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()  # 预测的轨迹索引
                }
                pred_dict_list.append(single_pred_dict)
            
            assert len(pred_dict_list) == batch_dict['batch_size']  # 确保预测数量与批次大小一致
            self.pred_dicts += pred_dict_list  # 将预测结果添加到总预测字典中
    
    def compute_metrics_and_log(self, batch, prediction, status='train'):
        """Compute metrics and log results"""
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']
        
        predicted_traj = prediction['predicted_trajectory']
        predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()
        
        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)
        
        # Calculate FDE losses
        bs, modes, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)
        
        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)
        
        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob = predicted_prob[np.arange(bs), best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + np.square(1 - predicted_prob)
        
        loss_dict = {
            'minADE6': np.mean(minade),
            'minFDE6': np.mean(minfde),
            'miss_rate': np.mean(miss_rate.astype(np.float32)),
            'brier_fde': np.mean(brier_fde)
        }
        
        return loss_dict
    
    def compute_metrics_waymo(self, pred_dicts):
        from models.base_model.waymo_eval import waymo_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts,
                                                             num_modes_for_eval=num_modes_for_eval)
        
        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str
        
        return metric_result_str, metric_results
    
    def compute_metrics_nuscenes(self, pred_dicts):
        from nuscenes.eval.prediction.compute_metrics import compute_metrics
        metric_results = compute_metrics(pred_dicts, self.helper, self.pred_config5)
        return metric_results
    
    def compute_metrics_av2(self, pred_dicts):
        from models.base_model.av2_eval import argoverse2_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results = argoverse2_evaluation(pred_dicts=pred_dicts,
                                               num_modes_for_eval=num_modes_for_eval)
        return metric_results
    
    def configure_optimizers(self):
        raise NotImplementedError
