"""
TrajAttr PyTorch 评估脚本

该脚本提供了用于评估轨迹预测模型的纯PyTorch实现。
支持TrajAttr框架中的所有模型：AutoBot、MTR、SMART和Wayformer。

使用方法:
    python evaluation_torch.py ckpt_path=path/to/checkpoint.ckpt
    python evaluation_torch.py method=autobot ckpt_path=path/to/checkpoint.ckpt
    python evaluation_torch.py eval_nuscenes=True ckpt_path=path/to/checkpoint.ckpt
"""

import os
import json
from pathlib import Path
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from models import build_model
from utils_datasets_traj import build_dataset
import utils_datasets_traj.common_utils as common_utils
from utils.utils_train_traj import set_seed

from utils.path_manager import path_manager

# 设置PyTorch精度以在现代GPU上获得更好的性能
torch.set_float32_matmul_precision('medium')


def move_to_device(batch, device):
    """递归地将batch数据移动到指定设备"""
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                move_to_device(value, device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch[key] = [v.to(device) for v in value]


def load_checkpoint(model, checkpoint_path, device):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点未找到: {checkpoint_path}")
    
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        # 自定义训练器格式
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"从第 {checkpoint.get('epoch', '未知')} 轮加载模型")
    elif 'state_dict' in checkpoint:
        # PyTorch Lightning格式
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 直接状态字典
        model.load_state_dict(checkpoint)
    
    print("模型权重加载成功")


def aggregate_metrics(metrics_list):
    """聚合批次指标"""
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated

def _stringify_identifier(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except Exception:
            return value.hex()
    return str(value)


def collect_prediction_records(batch_dict, prediction, max_modes=None):
    results = []
    if not isinstance(prediction, dict):
        return results

    input_dict = batch_dict.get('input_dict') if isinstance(batch_dict, dict) else None
    if input_dict is None:
        return results

    pred_scores = prediction.get('predicted_probability')
    pred_trajs = prediction.get('predicted_trajectory')
    if pred_scores is None or pred_trajs is None:
        return results

    if not hasattr(pred_trajs, 'detach'):
        pred_trajs = torch.as_tensor(pred_trajs)
    if not hasattr(pred_scores, 'detach'):
        pred_scores = torch.as_tensor(pred_scores, device=pred_trajs.device, dtype=pred_trajs.dtype)

    device = pred_trajs.device
    dtype = pred_trajs.dtype

    center_world = input_dict.get('center_objects_world')
    if center_world is None:
        return results
    if not isinstance(center_world, torch.Tensor):
        center_world = torch.as_tensor(center_world, device=device, dtype=dtype)
    else:
        center_world = center_world.to(device=device, dtype=dtype)

    map_center = input_dict.get('map_center')
    if map_center is None:
        return results
    if not isinstance(map_center, torch.Tensor):
        map_center = torch.as_tensor(map_center, device=device, dtype=dtype)
    else:
        map_center = map_center.to(device=device, dtype=dtype)

    center_world_np = center_world.detach().cpu().numpy()
    map_center_np = map_center.detach().cpu().numpy()

    num_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
    if num_objects == 0:
        return results

    reshaped = pred_trajs.reshape(num_objects, num_modes * num_timestamps, num_feat)
    angles = center_world[:, 6].reshape(num_objects)
    rotated = common_utils.rotate_points_along_z_tensor(points=reshaped, angle=angles)
    rotated = rotated.reshape(num_objects, num_modes, num_timestamps, num_feat)
    rotated[:, :, :, 0:2] += center_world[:, None, None, 0:2] + map_center[:, None, None, 0:2]

    pred_trajs_np = rotated[:, :, :, 0:2].detach().cpu().numpy()
    pred_scores_np = pred_scores.detach().cpu().numpy()

    scenario_ids = input_dict.get('scenario_id', [])
    object_ids = input_dict.get('center_objects_id', [])

    for idx in range(num_objects):
        probs = pred_scores_np[idx]
        safe_probs = np.nan_to_num(probs, nan=-np.inf)
        order = np.argsort(safe_probs)[::-1]
        if max_modes is not None:
            try:
                max_modes_int = int(max_modes)
                order = order[:max_modes_int]
            except (TypeError, ValueError):
                pass

        trajectories = []
        probabilities = []
        mode_indices = []
        current_point = [
            float(center_world_np[idx, 0] + map_center_np[idx, 0]),
            float(center_world_np[idx, 1] + map_center_np[idx, 1]),
        ]
        for mode_idx in order:
            future_points = pred_trajs_np[idx, mode_idx].tolist()
            trajectory = [list(current_point)] + future_points
            trajectories.append(trajectory)
            prob_value = float(probs[mode_idx]) if not math.isnan(probs[mode_idx]) else None
            probabilities.append(prob_value)
            mode_indices.append(int(mode_idx))

        results.append({
            'scenario_id': _stringify_identifier(scenario_ids[idx] if idx < len(scenario_ids) else None),
            'object_id': _stringify_identifier(object_ids[idx] if idx < len(object_ids) else None),
            'track_id': _stringify_identifier(object_ids[idx] if idx < len(object_ids) else None),
            'mode_index': mode_indices,
            'predicted_probability': probabilities,
            'predicted_trajectory': trajectories,
        })

    return results
    """聚合批次指标"""

    if not metrics_list:
        return {}

    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated


def compute_official_metrics(model, config):
    """计算官方评估指标"""
    if not hasattr(model, 'pred_dicts') or not model.pred_dicts:
        return {}
    
    official_metrics = {}
    
    try:
        # nuScenes评估
        if config.get('eval_nuscenes', False):
            print("计算nuScenes官方指标...")
            os.makedirs('submission', exist_ok=True)
            
            # 保存预测结果
            submission_path = os.path.join('submission', 'evalai_submission.json')
            with open(submission_path, 'w') as f:
                json.dump(model.pred_dicts, f)
            
            # 计算指标
            if hasattr(model, 'compute_metrics_nuscenes'):
                nuscenes_metrics = model.compute_metrics_nuscenes(model.pred_dicts)
                official_metrics['nuscenes'] = nuscenes_metrics
        
        # Waymo评估
        elif config.get('eval_waymo', False):
            print("计算Waymo官方指标...")
            if hasattr(model, 'compute_metrics_waymo'):
                waymo_metrics, result_str = model.compute_metrics_waymo(model.pred_dicts)
                official_metrics['waymo'] = waymo_metrics
                print("Waymo结果:")
                print(result_str)
        
        # Argoverse2评估
        elif config.get('eval_argoverse2', False):
            print("计算Argoverse2官方指标...")
            if hasattr(model, 'compute_metrics_av2'):
                av2_metrics = model.compute_metrics_av2(model.pred_dicts)
                official_metrics['argoverse2'] = av2_metrics
    
    except Exception as e:
        print(f"计算官方指标时出错: {e}")
    
    return official_metrics


def print_results(metrics, total_samples, avg_loss):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    print(f"总评估样本数: {total_samples}")
    print(f"平均损失: {avg_loss:.4f}")
    
    # 打印标准指标
    standard_metrics = ['min_ade', 'min_fde', 'miss_rate', 'brier_ade', 'brier_fde']
    print("\n标准指标:")
    for metric in standard_metrics:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")
    
    # 打印官方指标
    for dataset_name in ['nuscenes', 'waymo', 'argoverse2']:
        if dataset_name in metrics:
            print(f"\n{dataset_name.upper()} 官方指标:")
            dataset_metrics = metrics[dataset_name]
            if isinstance(dataset_metrics, dict):
                for key, value in dataset_metrics.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {dataset_metrics}")
    
    print("=" * 60 + "\n")


@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="config")
# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    """主评估函数"""
    # 设置随机种子以确保可重现性
    path_manager.resolve_config_paths(cfg)
    set_seed(cfg.seed)
    
    # 启用配置修改
    OmegaConf.set_struct(cfg, False)
    
    # 合并方法特定配置
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    export_prediction_max_modes = cfg.get('export_prediction_max_modes', None)
    export_prediction_path = cfg.get('export_prediction_path', None)
    if export_prediction_max_modes is not None:
        try:
            export_prediction_max_modes = int(export_prediction_max_modes)
        except (TypeError, ValueError):
            export_prediction_max_modes = None
    if export_prediction_path is not None:
        export_prediction_path = Path(export_prediction_path)
    prediction_records = []
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.debug else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型和数据集
    print(f"构建 {cfg.method.model_name} 模型...")
    model = build_model(cfg).to(device)
    
    # 加载检查点
    if cfg.ckpt_path:
        load_checkpoint(model, cfg.ckpt_path, device)
    else:
        print("警告: 未指定检查点路径。评估将使用随机权重。")
    
    # 设置验证数据集
    print("设置验证数据集...")
    val_dataset = build_dataset(cfg, val=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.method['eval_batch_size'],
        num_workers=cfg.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据集已加载: {len(val_dataset)} 个样本, {len(val_loader)} 个批次")
    
    # 开始评估
    print("开始评估...")
    model.eval()
    
    # 清除预测结果
    if hasattr(model, 'pred_dicts'):
        model.pred_dicts = []
    
    # 评估循环
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='评估中')):
            # 将数据移动到设备
            move_to_device(batch, device)
            
            # 前向传播
            try:
                prediction, loss = model(batch)
            except Exception as e:
                print(f"批次 {batch_idx} 出错: {e}")
                continue
            
            # 更新统计信息
            total_loss += loss.item()
            batch_count += 1
            
            # 计算样本数
            if hasattr(batch, 'batch_size'):
                total_samples += batch.batch_size
            elif 'batch_size' in batch:
                total_samples += batch['batch_size']
            else:
                total_samples += 1
            
            # 计算批次指标
            try:
                batch_metrics = model.compute_metrics_and_log(batch, prediction, status='val')
                all_metrics.append(batch_metrics)
            except Exception as e:
                print(f"计算指标时出错: {e}")
            
            # 存储预测结果用于官方评估
            if hasattr(model, 'compute_official_evaluation'):
                try:
                    model.compute_official_evaluation(batch, prediction)
                except Exception as e:
                    print(f"[Warning] official evaluation failed: {e}")
            prediction_records.extend(collect_prediction_records(batch, prediction, export_prediction_max_modes))
    aggregated_metrics = aggregate_metrics(all_metrics)
    avg_loss = total_loss / max(batch_count, 1)
    official_metrics = compute_official_metrics(model, cfg)
    if prediction_records:
        if export_prediction_path is not None:
            output_path = export_prediction_path
        else:
            model_name = 'model'
            if hasattr(cfg, 'method') and cfg.method is not None and 'model_name' in cfg.method:
                model_name = str(cfg.method.model_name)
            default_name = f'predictions_{model_name}_world_coords.json'
            output_path = Path(__file__).resolve().parent / default_name
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump({'predictions': prediction_records}, f, ensure_ascii=False, indent=2)
        print(f"Predictions exported to {output_path}")
    # 合并所有指标
    all_metrics_dict = {**aggregated_metrics, **official_metrics}
    
    # 打印结果
    print_results(all_metrics_dict, total_samples, avg_loss)
    
    # 可选：保存结果
    if cfg.get('save_results', False):
        results = {
            'metrics': all_metrics_dict,
            'total_samples': total_samples,
            'avg_loss': avg_loss
        }
        
        results_path = f"evaluation_results_{cfg.method.model_name}.json"
        with open(results_path, 'w') as f:
            # 转换numpy类型为Python类型
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        print(f"结果已保存至 {results_path}")
    
    return all_metrics_dict


if __name__ == '__main__':
    main()
