"""
轨迹预测归因评估独立启动脚本
exps_scripts/exp_trajattr/eval_traj_attr.py

功能：
1. 加载训练好的模型和验证数据集。
2. 从磁盘读取已保存的 numpy 格式归因结果。
3. 运行 AttributionEvaluator 计算归因指标 (MoRF, LeRF, Sen-n, etc.)。
4. 保存评估结果到 JSON 文件。

使用方法：
python exps_scripts/exp_trajattr/eval_traj_attr.py \
    attribution.load_path="/path/to/your/saved/attributions/numpy" \
    evaluate=True
"""

import sys
import os
import json
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

# 添加项目根目录到环境变量
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from models import build_model
from utils_datasets_traj import build_dataset
from utils.utils_train_traj import set_seed
from utils.path_manager import path_manager
# [注意] 这里根据您的要求修改了导入路径
from utils_attr.attr_metric.attr_evaluation import AttributionEvaluator

class TrajAttrEvalExperiment:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.get("debug", False) else "cpu"
        )
        set_seed(config.get("seed", 42))
        
        # 验证必要的配置
        self.attr_load_path = Path(config.attribution.get("load_path", ""))
        if not self.attr_load_path.exists():
            raise FileNotFoundError(f"归因结果路径不存在: {self.attr_load_path}\n请在命令行通过 attribution.load_path=... 指定")

        # 设置结果保存路径 (默认保存在归因加载目录的同级 metrics 目录下)
        self.save_dir = self.attr_load_path.parent / "metrics_evaluation"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"评估结果将保存至: {self.save_dir}")

        self.model = None
        self.val_loader = None
        self.evaluator = None
        self.eval_results_history = {}

    def load_model_and_data(self):
        """加载模型和数据"""
        print(f"正在加载模型: {self.config.method.model_name} ...")
        self.model = build_model(self.config).to(self.device)
        
        # 加载权重
        ckpt_path = self.config.get("ckpt_path")
        if ckpt_path and Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print("模型权重加载成功")
        else:
            print(f"警告: 未找到权重文件 {ckpt_path}，使用随机初始化 (可能会影响 MoRF/LeRF 评估)")

        self.model.eval()

        # 加载数据
        print("正在构建验证数据集...")
        val_dataset = build_dataset(self.config, val=True)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.method.get("eval_batch_size", 1),
            num_workers=self.config.get("load_num_workers", 0),
            shuffle=False,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"数据集大小: {len(val_dataset)} 样本, {len(self.val_loader)} 批次")

    def init_evaluator(self):
        """初始化评估器"""
        print("初始化归因评估器...")
        self.evaluator = AttributionEvaluator(self.model)

    def _find_attribution_file(self, batch, method_name):
        """
        [修改] 根据 batch 中的 scenario_id 查找对应的 .npy 文件
        文件名示例: scene_00a8dc97-401a-4456-8f8c-f2dfbecbe343_AttnLRP_map_polylines.npy
        """
        # 1. 从 batch 中提取 scenario_id
        try:
            input_dict = batch.get('input_dict', {})
            scenario_ids = input_dict.get('scenario_id', [])
            
            if len(scenario_ids) == 0:
                print(f"[Warning] 当前 Batch 中没有找到 scenario_id")
                return None
            
            # 取第一个样本的 ID (假设 eval_batch_size=1)
            # 如果是 Tensor 转 string，如果是 list 直接取
            target_id = scenario_ids[0]
            if isinstance(target_id, torch.Tensor):
                target_id = str(target_id.item())
            else:
                target_id = str(target_id)
                
        except Exception as e:
            print(f"[Error] 解析 scenario_id 失败: {e}")
            return None

        # 2. 并在目录下搜索包含该 ID 的文件
        # 文件名通常包含: scenario_id 和 method_name
        # 且我们需要 'obj_trajs' 的文件来进行评估
        target_files = []
        
        # 遍历目录 (为了性能，建议实际部署时先建立索引，这里用简单遍历)
        for f in self.attr_load_path.glob("*.npy"):
            fname = f.name
            
            # 核心匹配逻辑：文件名必须包含 ID 和 方法名
            if target_id in fname and method_name in fname:
                target_files.append(f)
        
        # 3. 筛选出 obj_trajs 文件 (评估重点)
        obj_traj_files = [f for f in target_files if 'obj_trajs' in f.name]
        
        if obj_traj_files:
            # print(f"[Debug] Found file: {obj_traj_files[0].name}") 
            return obj_traj_files[0]
        
        # 如果没找到 obj_trajs 但有其他文件(比如 map)，尝试返回(可能导致 key 错误)
        # 或者直接返回 None
        if target_files:
            # print(f"[Debug] Found related file (not obj_trajs): {target_files[0].name}")
            return target_files[0]
            
        print(f"[Warning] 未找到 ID={target_id}, Method={method_name} 的 obj_trajs 归因文件")
        return None

    def run(self):
        self.load_model_and_data()
        self.init_evaluator()

        methods = self.config.attribution.methods
        metrics_to_compute = self.config.get("eval_metrics", ['morf', 'lerf', 'sen_n', 'sparseness', 'complexity'])
        print(f"将要评估的方法: {methods}")
        print(f"将要计算的指标: {metrics_to_compute}")

        batch_limit = self.config.attribution.get("batch_limit", 999999)
        
        # 开始循环评估
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
            if batch_idx >= batch_limit:
                break
            
            # 将数据移动到设备
            batch = self._move_to_device(batch, self.device)
            
            # 对每个指定的归因方法进行评估
            for method_name in methods:
                # [修改关键点] 传入完整 batch，以便内部提取 scenario_id
                attr_file = self._find_attribution_file(batch, method_name)
                
                if attr_file is None:
                    # 如果没找到文件，已经在 _find_attribution_file 里打印了 Warning，这里直接跳过
                    continue
                
                try:
                    # 1. 加载 numpy 数据
                    attr_numpy = np.load(attr_file)
                    
                    # [新增] 维度检查与适配
                    # 某些保存逻辑可能保存为 [Num_Agents, Time, Feat] (缺少 Batch 维)
                    # 评估器期望 [Batch, Num_Agents, Time, Feat]
                    if attr_numpy.ndim == 3:
                        attr_numpy = attr_numpy[np.newaxis, ...]
                    
                    # 构造 attributions 字典 (评估器需要这个格式)
                    # 默认加载的是 obj_trajs 的文件
                    attributions = {
                        'obj_trajs': torch.from_numpy(attr_numpy).to(self.device)
                    }
                    
                    # 2. 运行评估
                    metrics_result = self.evaluator.evaluate(
                        batch=batch,
                        attributions=attributions,
                        metrics=metrics_to_compute,
                        target_key='obj_trajs',
                        steps=10,
                        num_subsets=20
                    )
                    
                    # 3. 记录结果
                    if method_name not in self.eval_results_history:
                        self.eval_results_history[method_name] = []
                    self.eval_results_history[method_name].append(metrics_result)
                    
                except Exception as e:
                    print(f"[Error] 评估 Batch {batch_idx} (Method: {method_name}) 时出错: {e}")
                    import traceback
                    traceback.print_exc()

        # 所有批次结束后，保存最终结果
        self._save_results()

    def _save_results(self):
        save_path = self.save_dir / "offline_evaluation_metrics.json"
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)): return int(obj)
            if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
            return obj

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(self.eval_results_history), f, indent=2, ensure_ascii=False)
        
        print(f"\n[完成] 评估结束，结果已保存至: {save_path}")

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {key: self._move_to_device(value, device) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_to_device(item, device) for item in batch]
        return batch

@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="traj_attr_base")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # 确保合并了方法的具体配置 (如 autobot.yaml)
    if hasattr(cfg, 'method'):
        cfg = OmegaConf.merge(cfg, cfg.method)
    
    experiment = TrajAttrEvalExperiment(cfg)
    experiment.run()

if __name__ == "__main__":
    main()