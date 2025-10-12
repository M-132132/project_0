"""
轨迹预测归因计算脚本

运行模型、计算归因，并将归因结果保存为 `.npy` 文件。
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models import build_model
from utils_datasets_traj import build_dataset
from utils.utils_train_traj import set_seed
from utils.path_manager import path_manager
from utils_attr.traj_attr.base.traj_attr_base import TrajAttrBase


@hydra.main(version_base=None, config_path=str(path_manager.get_config_path()), config_name="traj_attr_base")
def main(cfg: DictConfig) -> None:
    """主函数：执行轨迹预测归因实验。"""
    OmegaConf.set_struct(cfg, False)

    # 合并配置文件，将基础配置(cfg)与方法特定配置(cfg.method)合并为运行时配置(runtime_cfg)
    runtime_cfg = OmegaConf.merge(cfg, cfg.method)

    # 创建轨迹属性实验实例，使用合并后的运行时配置
    experiment = TrajAttrExperiment(runtime_cfg)
    # 执行归因实验并获取处理后的批次信息
    processed_batches = experiment.run_attribution_experiment()

    # 打印实验完成信息
    print("\n[OK] 归因实验完成")
    # 打印已处理的批次数量
    print(f"处理批次: {processed_batches}")
    # 打印归因结果保存的目录路径
    print(f"归因目录: {experiment.paths['numpy']}")


class TrajAttrExperiment:
    def __init__(self, config: DictConfig):
        """
        初始化方法，用于设置模型的配置、设备、随机种子和路径等

        参数:
            config (DictConfig): 包含模型配置的字典对象，包括设备选择、随机种子等参数
        """
    # 保存传入的配置对象，以便后续使用
        self.config = config
    # 设置计算设备：如果有可用的CUDA且不在调试模式下，则使用GPU；否则使用CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.get("debug", False) else "cpu"
        )

    # 设置随机种子以确保实验可复现性，默认值为42
        set_seed(config.get("seed", 42))
    # 设置保存结果的路径，调用_setup_save_paths方法创建相关目录
        self.paths = self._setup_save_paths()

    # 初始化模型验证加载器和归因器的变量，这些变量将在后续方法中实例化
        self.model = None
        self.val_loader = None
        self.attributor = None

    def _setup_save_paths(self) -> Dict[str, Path]:
        """
        设置保存路径的函数，根据配置创建实验目录结构
        Returns:
            Dict[str, Path]: 包含不同保存路径的字典，包括基础目录、归因目录和numpy格式保存目录
        该函数会:
        1. 解析基础目录路径
        2. 获取方法和数据集名称
        3. 根据配置决定是否创建时间戳目录
        4. 创建必要的目录结构
        5. 打印输出目录信息
        """
    # 解析基础目录路径
        base_dir = path_manager.resolve_path(self.config.save_config.base_dir)
    # 从配置中获取模型名称，默认为"unknown"
        method_name = self.config.method.get("model_name", "unknown")
    # 从配置中获取数据集名称，默认为"dataset"
        dataset_name = self.config.get("dataset_name", "dataset")

    # 根据配置决定是否创建带时间戳的目录
        if self.config.save_config.get("create_timestamp_dir", False):
        # 生成当前时间戳，格式为年月日_时分秒
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建包含时间戳的实验目录路径
            exp_dir = Path(base_dir) / f"{method_name}_{dataset_name}" / timestamp
        else:
        # 创建不包含时间戳的实验目录路径
            exp_dir = Path(base_dir) / f"{method_name}_{dataset_name}"

    # 定义需要创建的目录结构字典
        paths = {
            "base": exp_dir,           # 基础实验目录
            "attributions": exp_dir / "attributions",  # 归因结果目录
            "numpy": exp_dir / "attributions" / "numpy",  # numpy格式归因结果目录
        }

    # 创建所有必要的目录（包括父目录），如果目录已存在则不创建
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        print(f"输出目录: {paths['base']}")
        return paths

    def load_model_and_data(self) -> None:
    # 使用多行注释说明函数功能
        """
        加载模型和验证数据的函数
        1. 根据配置构建模型
        2. 尝试加载预训练权重
        3. 构建验证数据加载器
        """
    # 获取模型名称并打印加载信息
        method_name = self.config.method.get("model_name", "unknown")
        print(f"加载模型: {method_name}")

    # 根据配置构建模型并将其移动到指定设备
        self.model = build_model(self.config).to(self.device)

    # 获取检查点路径并尝试加载权重
        ckpt_path = self.config.get("ckpt_path")
        if ckpt_path:
        # 解析路径并检查文件是否存在
            resolved_path = path_manager.resolve_path(ckpt_path)
            if Path(resolved_path).exists():
            # 加载检查点文件
                checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)
            # 尝试获取模型状态字典，支持多种可能的键名
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
            # 加载权重到模型
                self.model.load_state_dict(state_dict)
                print("模型权重加载成功")
            else:
            # 文件不存在时的警告信息
                print(f"警告: 未找到权重 {resolved_path}，使用随机初始化")

    # 加载验证数据集
        print("加载验证数据中...")
        val_dataset = build_dataset(self.config, val=True)
    # 创建验证数据加载器
        self.val_loader = DataLoader(
            val_dataset,
        # 获取评估批次大小，默认为1
            batch_size=self.config.method.get("eval_batch_size", 1),
        # 获取数据加载的工作进程数，默认为0
            num_workers=self.config.get("load_num_workers", 0),
        # 不打乱数据顺序
            shuffle=False,
        # 不丢弃最后一个不完整的批次
            drop_last=False,
        # 使用数据集的自定义批处理函数
            collate_fn=val_dataset.collate_fn,
        # 如果使用CUDA，则固定内存
            pin_memory=torch.cuda.is_available(),
        )
    # 打印数据集和批次信息
        print(f"数据集大小: {len(val_dataset)} 样本, {len(self.val_loader)} 批次")

    def create_attributor(self) -> None:
        """
        创建归因计算器的方法
        该方法根据配置中的模型名称创建相应的归因计算器实例。
        它从配置中获取模型名称，并使用该名称创建TrajAttrBase类的实例。
        参数:
            self: 类实例
        返回:
            None: 该方法不返回任何值
        """
    # 从配置中获取模型名称，如果未指定则使用默认值"unknown"
        method_name = self.config.method.get("model_name", "unknown")
    # 创建归因计算器实例，传入模型、配置和保存路径
        self.attributor = TrajAttrBase(self.model, self.config, save_paths=self.paths)
    # 打印创建的归因计算器信息
        print(f"创建归因计算器: {method_name}")

    def run_attribution_experiment(self) -> int:

        """
        运行归因实验的主函数
        该函数负责加载模型和数据、创建归因器、设置模型为评估模式，并计算批次归因

        返回:
            int: 批次归因计算的结果
        """
        self.load_model_and_data()  # 加载模型和数据
        self.create_attributor()    # 创建归因器
        self.model.eval()           # 将模型设置为评估模式，关闭dropout等训练时使用的层

        return self.compute_batch_attributions()  # 计算并返回批次归因结果

    def compute_batch_attributions(self) -> int:

        """
        计算并保存一批样本的归因分析结果

        该方法遍历验证数据加载器，对每个批次计算归因分析，直到达到配置的批次限制。
        使用tqdm进度条显示处理进度，并将数据移动到指定设备上进行计算。

        Returns:
            int: 已处理的批次数量
        """
        batch_limit = self.config.attribution.batch_limit  # 获取配置的批次限制数量
        processed = 0  # 初始化已处理批次计数器

    # 使用tqdm创建进度条，遍历验证数据加载器
        for batch_idx, batch in enumerate(
            tqdm(self.val_loader, desc="Computing attributions", dynamic_ncols=True)
        ):
        # 如果当前批次索引超过配置的限制，则停止处理
            if batch_idx >= batch_limit:
                break

        # 将批次数据移动到指定设备（如GPU）
            batch = self._move_to_device(batch, self.device)

        # 启用梯度计算，以便进行归因分析
            with torch.enable_grad():
            # 计算并保存当前批次的归因分析结果
                self.attributor.compute_and_save_attribution(
                    batch,
                    methods=self.config.attribution.methods,
                )

            processed += 1  # 增加已处理批次计数

        return processed  # 返回已处理的批次总数

    def _move_to_device(self, batch, device):
        """
        将数据移动到指定设备（如GPU）的递归函数
        参数:
            batch: 需要移动的数据，可以是张量、字典或列表
            device: 目标设备，如 'cuda' 或 'cpu'
        返回:
            移动到目标设备后的数据，保持原始数据结构
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)  # 如果是张量，直接移动到指定设备
        if isinstance(batch, dict):
            # 如果是字典，递归处理每个值，保持键不变
            return {key: self._move_to_device(value, device) for key, value in batch.items()}
        if isinstance(batch, list):
            # 如果是列表，递归处理每个元素，保持列表结构
            return [self._move_to_device(item, device) for item in batch]
        return batch  # 其他类型数据直接返回，不做处理


if __name__ == "__main__":
    main()
