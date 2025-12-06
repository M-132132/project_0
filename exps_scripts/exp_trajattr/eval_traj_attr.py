"""
轨迹预测归因评估独立启动脚本
exps_scripts/exp_trajattr/eval_traj_attr.py

功能：
1. 加载训练好的模型和验证数据集。
2. 从磁盘读取已保存的 numpy 格式归因结果 (支持按 Batch 堆叠)。
3. 运行 AttributionEvaluator 计算归因指标 (支持 Agent级/Feature级, Loss/Shift 指标)。
4. 保存评估结果到 JSON 文件。

使用方法示例：
1. 传统整车评估:
   python exps_scripts/exp_trajattr/eval_traj_attr.py \
       attribution.load_path="exps_res/.../numpy" \
       eval_mode="agent" metric_type="loss"

2. 细粒度特征扰动评估 (推荐):
   python exps_scripts/exp_trajattr/eval_traj_attr.py \
       attribution.load_path="exps_res/.../numpy" \
       eval_mode="feature" metric_type="shift" noise_std=1.0
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

        # 设置结果保存路径 (默认保存在归因加载目录的同级 metrics_evaluation 目录下)
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
            print(f"警告: 未找到权重文件 {ckpt_path}，使用随机初始化 (可能会影响评估结果)")

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

    def _get_file_path(self, scenario_id, method_name):
        """
        根据 scenario_id 精确查找文件
        假设保存格式为: scene_{id}_{method}_obj_trajs.npy
        """
        # 兼容 Tensor 类型 ID
        sid_str = str(scenario_id.item()) if isinstance(scenario_id, torch.Tensor) else str(scenario_id)
        
        # 1. 尝试标准格式 (优先)
        fname = f"scene_{sid_str}_{method_name}_obj_trajs.npy"
        fpath = self.attr_load_path / fname
        if fpath.exists():
            return fpath
            
        # 2. 备选：尝试不带 'scene_' 前缀 (视 compute_traj_attr.py 保存逻辑而定)
        fname_alt = f"{sid_str}_{method_name}_obj_trajs.npy"
        fpath_alt = self.attr_load_path / fname_alt
        if fpath_alt.exists():
            return fpath_alt

        return None

    def _load_batch_attributions(self, batch, method_name):
        """
        加载整个 Batch 的归因结果并堆叠为张量
        Returns:
            np.ndarray: [Batch_Size, Num_Agents, Time, Feat]
        """
        input_dict = batch.get('input_dict', {})
        scenario_ids = input_dict.get('scenario_id', [])
        
        attr_list = []
        
        for sc_id in scenario_ids:
            fpath = self._get_file_path(sc_id, method_name)
            
            if fpath is None:
                # 如果缺少文件，返回 None 跳过整个 Batch (保证数据对齐)
                # 您也可以选择在这里打印 ID 以便调试
                return None
                
            try:
                # 加载单个样本 [N, T, F]
                attr = np.load(fpath)
                
                # 某些保存逻辑可能会多带一个 batch 维度 [1, N, T, F]，需要 squeeze
                if attr.ndim == 4 and attr.shape[0] == 1:
                    attr = attr[0]
                    
                attr_list.append(attr)
            except Exception as e:
                print(f"[Error] 加载文件失败 {fpath}: {e}")
                return None

        if not attr_list:
            return None

        # 堆叠
        try:
            return np.stack(attr_list, axis=0)
        except ValueError as e:
            print(f"[Error] 堆叠失败 (可能样本间维度不一致): {e}")
            return None

    def run(self):
        self.load_model_and_data()
        self.init_evaluator()

        methods = self.config.attribution.methods
        metrics_to_compute = self.config.get("eval_metrics", ['morf', 'lerf', 'sen_n', 'sparseness', 'complexity'])
        
        # [新增] 获取高级评估配置
        eval_mode = self.config.get("eval_mode", "agent")   # 'agent' (整车) 或 'feature' (细粒度)
        noise_std = self.config.get("noise_std", 1.0)       # 扰动噪声标准差 (仅在 feature 模式下有用)
        metric_type = self.config.get("metric_type", "loss") # 'loss' (NLL) 或 'shift' (预测位移)
        perturb_mode = self.config.get("perturb_mode", "freeze") # 扰动模式

        print(f"评估方法: {methods}")
        print(f"计算指标: {metrics_to_compute}")
        print(f"评估模式 (Mode): {eval_mode}")
        print(f"指标类型 (Type): {metric_type}")
        print(f"扰动模式 (Perturb): {perturb_mode}")
        if eval_mode == 'feature' and perturb_mode != 'freeze':
            print(f"特征扰动噪声 (Noise Std): {noise_std} m")

        batch_limit = self.config.attribution.get("batch_limit", 999999)
        print(f"Batch Limit: {batch_limit}")

        # 开始循环评估
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
            if batch_idx >= batch_limit:
                print(f"已达到 Batch 限制 ({batch_limit})，停止评估。")
                break
            
            # 将数据移动到设备
            batch = self._move_to_device(batch, self.device)
            
            # 对每个指定的归因方法进行评估
            for method_name in methods:
                # 1. 加载整个 Batch 的归因数据
                attr_batch_np = self._load_batch_attributions(batch, method_name)
                
                if attr_batch_np is None:
                    # 文件不全，跳过
                    continue
                
                try:
                    # 构造 attributions 字典
                    attributions = {
                        'obj_trajs': torch.from_numpy(attr_batch_np).to(self.device)
                    }
                    
                    # 2. 运行评估 (传递新的参数)
                    metrics_result = self.evaluator.evaluate(
                        batch=batch,
                        attributions=attributions,
                        metrics=metrics_to_compute,
                        target_key='obj_trajs',
                        steps=10,
                        num_subsets=20,
                        # [传递新参数]
                        evaluation_mode=eval_mode,
                        noise_std=noise_std,
                        metric_type=metric_type,
                        perturb_mode=perturb_mode
                    )
                    
                    # 3. 记录结果
                    if method_name not in self.eval_results_history:
                        self.eval_results_history[method_name] = []
                    self.eval_results_history[method_name].append(metrics_result)
                    
                except Exception as e:
                    print(f"[Error] Batch {batch_idx} Method {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # 保存最终结果
        self._save_results()

    def _save_results(self):
        # [修改] 文件名命名规则：res_metric_eval_{metric_type}
        metric_type = self.config.get("metric_type", "loss")
        filename = f"res_metric_eval_{metric_type}.json"
        
        save_path = self.save_dir / filename
        
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
    # 确保合并了方法的具体配置
    if hasattr(cfg, 'method'):
        cfg = OmegaConf.merge(cfg, cfg.method)
    
    experiment = TrajAttrEvalExperiment(cfg)
    experiment.run()

if __name__ == "__main__":
    main()