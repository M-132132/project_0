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
        
        self.attr_load_path = Path(config.attribution.get("load_path", ""))
        if not self.attr_load_path.exists():
            raise FileNotFoundError(f"归因结果路径不存在: {self.attr_load_path}\n请指定 attribution.load_path")

        self.save_dir = self.attr_load_path.parent / "metrics_evaluation"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"评估结果将保存至: {self.save_dir}")

        self.model = None
        self.val_loader = None
        self.evaluator = None
        self.eval_results_history = {}

    def load_model_and_data(self):
        print(f"正在加载模型: {self.config.method.model_name} ...")
        self.model = build_model(self.config).to(self.device)
        
        ckpt_path = self.config.get("ckpt_path")
        if ckpt_path and Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print("模型权重加载成功")
        else:
            print(f"警告: 未找到权重文件 {ckpt_path}")

        self.model.eval()

        print("正在构建验证数据集...")
        val_dataset = build_dataset(self.config, val=True)
        # 注意：如果显存不足，可以在这里强制 batch_size=1
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
        print("初始化归因评估器...")
        self.evaluator = AttributionEvaluator(self.model)

    def _get_file_path(self, scenario_id, method_name):
        """
        [修复] 精确构造文件名
        文件名格式通常为: scene_{id}_{method}_obj_trajs.npy
        """
        scenario_id_str = str(scenario_id)
        
        # 1. 尝试标准命名格式 (compute_traj_attr.py 的默认格式)
        fname = f"scene_{scenario_id_str}_{method_name}_obj_trajs.npy"
        fpath = self.attr_load_path / fname
        if fpath.exists():
            return fpath
            
        # 2. 备选：尝试不带 'scene_' 前缀或其他变体 (视具体保存逻辑而定)
        # 这里为了保险，也可以加一个模糊搜索，但必须确保 ID 精确
        # 为了性能，建议保持上面的精确匹配。如果找不到，说明保存命名有问题。
        return None

    def _load_batch_attributions(self, batch, method_name):
        """
        [新增] 加载整个 Batch 的归因结果并堆叠
        """
        input_dict = batch.get('input_dict', {})
        scenario_ids = input_dict.get('scenario_id', [])
        
        attr_list = []
        valid_indices = [] # 记录成功加载的索引，万一有文件缺失
        
        for idx, sc_id in enumerate(scenario_ids):
            if isinstance(sc_id, torch.Tensor):
                sc_id = sc_id.item()
                
            fpath = self._get_file_path(sc_id, method_name)
            
            if fpath is None:
                # 找不到文件，打印警告 (只打一次避免刷屏)
                if idx == 0: 
                    print(f"[Warning] 找不到文件: ID={sc_id}, Method={method_name}")
                continue
                
            try:
                # 加载单个样本归因 [N, T, F]
                attr_np = np.load(fpath)
                attr_list.append(attr_np)
                valid_indices.append(idx)
            except Exception as e:
                print(f"[Error] 加载文件失败 {fpath}: {e}")

        if not attr_list:
            return None

        # 堆叠成 Batch: [B_loaded, N, T, F]
        # 注意：如果 Batch 里有文件缺失，这里的 Batch 维度会小于原始 Batch
        # 严格来说应该报错或填充 0，但这里我们假设文件都是齐的
        try:
            batch_attr_np = np.stack(attr_list, axis=0) 
        except ValueError as e:
            print(f"[Error] 堆叠归因数组失败 (可能形状不一致): {e}")
            return None
            
        # 如果有文件缺失，我们需要对齐 batch 数据（这比较复杂）。
        # 这里简单起见：如果找到的数量 != batch size，建议跳过该 batch 以免指标计算错位
        if len(attr_list) != len(scenario_ids):
            print(f"[Warning] Batch 文件不全 ({len(attr_list)}/{len(scenario_ids)})，跳过此 Batch")
            return None

        return batch_attr_np

    def run(self):
        self.load_model_and_data()
        self.init_evaluator()

        methods = self.config.attribution.methods
        metrics_to_compute = self.config.get("eval_metrics", ['morf', 'lerf', 'sen_n', 'sparseness', 'complexity'])
        print(f"将要评估的方法: {methods}")
        print(f"将要计算的指标: {metrics_to_compute}")

        # [修改 1] 获取配置中的 limit，默认跑完全部 (999999)
        # 您可以在命令行使用 attribution.batch_limit=12 来控制
        limit = self.config.attribution.get("batch_limit", 999999) 
        print(f"评估 Batch 限制: {limit}")

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
            # [修改 2] 达到限制时停止
            if batch_idx >= limit:
                print(f"已达到 Batch 限制 ({limit})，停止评估。")
                break
            
            batch = self._move_to_device(batch, self.device)
            
            for method_name in methods:
                # 加载整个 Batch 的归因数据
                attr_batch_np = self._load_batch_attributions(batch, method_name)
                
                # 如果因为文件缺失导致返回 None，则跳过
                if attr_batch_np is None:
                    continue
                
                try:
                    attributions = {
                        'obj_trajs': torch.from_numpy(attr_batch_np).to(self.device)
                    }
                    
                    metrics_result = self.evaluator.evaluate(
                        batch=batch,
                        attributions=attributions,
                        metrics=metrics_to_compute,
                        target_key='obj_trajs',
                        steps=10, 
                        num_subsets=20
                    )
                    
                    if method_name not in self.eval_results_history:
                        self.eval_results_history[method_name] = []
                    self.eval_results_history[method_name].append(metrics_result)
                    
                except Exception as e:
                    print(f"[Error] Batch {batch_idx} Method {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

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
        if isinstance(batch, torch.Tensor): return batch.to(device)
        if isinstance(batch, dict): return {k: self._move_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, list): return [self._move_to_device(i, device) for i in batch]
        return batch

@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="traj_attr_base")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg, 'method'):
        cfg = OmegaConf.merge(cfg, cfg.method)
    experiment = TrajAttrEvalExperiment(cfg)
    experiment.run()

if __name__ == "__main__":
    main()