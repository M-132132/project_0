"""
AttnLRP 归因（Autobot 版，薄适配）

目标：
- 标量：每样本 minADE（保持 batch 维度）
- 对象：'obj_trajs' 与 'map_polylines'（从 .grad 读取 relevance）
- 规则：模块级替换（Linear/LayerNorm/MHA），首版使用 CP‑LRP 注意力

说明：
- 不改动模型源码；仅在归因阶段注册规则并反传 LRP 相关性
- 不依赖 FX 追踪；后续如需函数级替换可在 Composite.register 传入 tracer+dummy_inputs
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from utils_attr.attn_lrp.core import Composite
from utils_attr.attn_lrp.modules import (
    LinearEpsilon,
    LayerNormEpsilon,
    MultiheadAttention_CP,
    MultiheadAttention_AttnLRP,
)


class AttnLRPAttribution:
    """
    与 TrajAttrBase 接口一致的 AttnLRP 归因类。
    - 仅在首次归因时对模型做模块级规则替换
    - 前向得到每样本 minADE（保持 batch 维度），用全 1 上游梯度触发 LRP 反传
    - 从归因输入的 .grad 读取相关性并返回
    """

    def __init__(self, attr_base, epsilon: float = 1e-6, attention_rule: str = "CP", **kwargs):
        self.attr_base = attr_base
        self.model: nn.Module = attr_base.model
        self.device = attr_base.device
        self.epsilon = float(epsilon)
        self.attention_rule = str(attention_rule).upper()
        self._composite = None
        self._registered = False

    def _build_layer_map(self) -> Dict:
        """构建“模块 → 规则”的映射（首版使用 CP‑LRP 注意力）。"""
        attn_cls = MultiheadAttention_CP if self.attention_rule == "CP" else MultiheadAttention_AttnLRP
        return {
            nn.Linear: LinearEpsilon,           # 线性层 ε‑LRP
            nn.LayerNorm: LayerNormEpsilon,     # LayerNorm 混合规则
            nn.MultiheadAttention: attn_cls,    # 注意力：CP 或 AttnLRP
        }

    def _ensure_registered(self) -> None:
        """仅在第一次归因时注册模块级 LRP 规则（不做 FX 函数级替换）。"""
        if self._registered:
            return
        layer_map = self._build_layer_map()
        self._composite = Composite(layer_map=layer_map)
        # 模块级替换无需 dummy_inputs；函数级替换可后续加
        self._composite.register(self.model, dummy_inputs=None, verbose=False, no_grad=True)
        # 贯通 epsilon 到替换后的模块
        try:
            for m in self.model.modules():
                if isinstance(m, LinearEpsilon):
                    m.epsilon = self.epsilon
                # 注意力模块的 epsilon（CP/AttnLRP）
                if isinstance(m, (MultiheadAttention_CP, MultiheadAttention_AttnLRP)):
                    m.epsilon = self.epsilon
        except Exception:
            pass
        self._registered = True

    def compute_attribution(
        self,
        attribution_inputs: Dict[str, torch.Tensor],  # 归因输入，包含需要计算梯度的张量
        static_inputs: Dict[str, torch.Tensor],      # 静态输入，在计算过程中保持不变
        input_tensors: Tuple[torch.Tensor, ...],    # 输入张量元组
    ) -> Dict[str, torch.Tensor]:                  # 返回包含归因结果的字典
        """
        计算输入特征的归因分数，使用LRP（Layer-wise Relevance Propagation）方法。
        该方法通过以下步骤计算归因：
        1. 确保所有必要的规则都已注册
        2. 准备需要计算梯度的输入张量
        3. 前向传播计算每样本的损失
        4. 反向传播计算梯度
        5. 提取特定输入特征的归因分数
        6. 恢复模型的原始训练状态
        参数:
            attribution_inputs: 需要计算归因的输入字典
            static_inputs: 在计算过程中保持不变的静态输入字典
            input_tensors: 输入张量元组
        返回:
            包含特定输入特征归因分数的字典
        异常:
            RuntimeError: 当每样本损失维度不正确或特定特征的梯度为空时抛出
        """
        # 1) 注册规则
        self._ensure_registered()  # 确保所有必要的规则都已注册

        # 2) 切换 eval，准备归因输入（需梯度）
        prev_training = self.model.training  # 保存模型当前训练状态
        self.model.eval()  # 将模型切换为评估模式
    # 准备归因输入字典，确保需要计算梯度的张量被正确处理
        prepared_inputs: Dict[str, torch.Tensor] = {}
        for k, v in attribution_inputs.items():
            if isinstance(v, torch.Tensor):
            # 克隆张量并启用梯度计算
                prepared_inputs[k] = v.detach().clone().requires_grad_(True).to(self.device)
        # 其余保持原引用（若需）
        for k, v in attribution_inputs.items():
            if k not in prepared_inputs:
                prepared_inputs[k] = v

        # 3) 前向得到每样本 minADE（[B] 向量）
        self.model.zero_grad(set_to_none=True)
        per_sample_loss = self.attr_base.model_forward_wrapper(
            prepared_inputs, static_inputs, target_trajs=None
        )
        if per_sample_loss.ndim != 1:
            # 尽量给出可读错误，方便排查
            raise RuntimeError("AttnLRP 期望每样本标量向量 (B,) 作为初始相关性，请检查 distance_type 是否为 batch-preserved 版本。")

        # 4) 触发 LRP 反传：对 (B,) 向量用全 1 的上游梯度
        grad_out = torch.ones_like(per_sample_loss, device=self.device)
        per_sample_loss.backward(grad_out)

        # 5) 从输入 .grad 读取相关性
        attributions: Dict[str, torch.Tensor] = {}
        for key in ["obj_trajs", "map_polylines"]:
            t = prepared_inputs.get(key, None)
            if isinstance(t, torch.Tensor):
                g = t.grad
                if g is None:
                    raise RuntimeError(f"{key} 未得到 LRP 相关性（.grad 为空）")
                attributions[key] = g.detach()

        # 6) 恢复训练状态
        if prev_training:
            self.model.train()

        return attributions
