import math
import torch
import torch.fx
import torch.nn.functional as F
# 修改为相对导入：rules 使用本目录实现
from . import rules


@torch.no_grad()
def _prepare_key_padding_mask(mask, attn_mask, query):
    """
    为注意力机制操作准备键填充掩码。
    """
    # -- 广播掩码
    assert mask.ndim > 1 # [..., SeqLen]
    if mask.ndim == 2: # [Batch, ... , ... , SeqLen]
        b, k_len = mask.shape
        mask = mask.view(b, 1, 1, k_len)

    return F._canonical_mask(mask, "key_padding_mask", F._none_or_dtype(attn_mask), "attn_mask", query.dtype)

@torch.no_grad()
def _prepare_attn_mask(mask, query):
    """
    为注意力机制操作准备注意力掩码。
    """
    # -- 广播掩码
    assert mask.ndim >= 2 # [..., SeqLen, SeqLen]
    if mask.ndim == 3: # [Batch * Heads, SeqLen, SeqLen]
        mask = mask.view(query.shape)

    return F._canonical_mask(mask, "attn_mask", None, "", query.dtype, False)

# 引入函数式 LRP 实现（用于 softmax 等函数的 LRP 反传）
from . import functional as lf

@torch.fx.wrap
def multi_head_attention_cp(query, key, value, batch_first, num_heads, head_dim, q_proj_weight, bias_q, k_proj_weight, bias_k, v_proj, out_proj,
                            key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, epsilon: float = 1e-6):
    """
    为注意力机制实现CP-LRP（保守传播-LRP）规则，即我们不让相关性通过softmax流动，而只通过值路径流动。
    此方法*仅在Vision Transformers中效果良好*，因为在这里高级的AttnLRP规则（确实使用softmax）与CP-LRP规则具有相似的性能。
    AttnLRP的问题在于使用softmax会引入梯度破碎，这需要应用z-plus LRP规则。
    这使得AttnLRP效率略低，根据我们有限的实验，在Vision Transformers中小的性能提升并不值得。
    然而，在大型语言模型中，在softmax上应用AttnLRP明显优于CP-LRP，且不需要效率较低的z-plus规则。
    因此，我们为注意力机制选择更高效的CP-LRP，并为ViT的其他部分使用AttnLRP。

    请参考论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'
    的A.2.3节'Tackling Noise in Vision Transformers'。

    参数:
    -----------
    query: torch.Tensor
        查询张量，如果batch_first为False，形状为[SeqLen, Batch, Embed]，否则为[Batch, SeqLen, Embed]
    key: torch.Tensor
        键张量，如果batch_first为False，形状为[SeqLen, Batch, Embed]，否则为[Batch, SeqLen, Embed]
    value: torch.Tensor
        值张量，如果batch_first为False，形状为[SeqLen, Batch, Embed]，否则为[Batch, SeqLen, Embed]
    batch_first: bool
        输入张量是否采用batch_first格式
    num_heads: int
        注意力头的数量
    head_dim: int
        每个注意力头的维度
    q_proj_weight: torch.Tensor
        查询张量的投影权重
    bias_q: torch.Tensor
        查询张量的偏置
    k_proj_weight: torch.Tensor
        键张量的投影权重
    bias_k: torch.Tensor
        键张量的偏置
    v_proj: torch.nn.Module
        值张量的投影模块
    out_proj: torch.nn.Module
        输出张量的投影模块
    key_padding_mask: torch.Tensor
        键张量的填充掩码
    need_weights: bool
        是否返回注意力权重
    attn_mask: torch.Tensor
        注意力掩码
    average_attn_weights: bool
        是否对注意力权重求平均
    
    返回:
    --------
    out: torch.Tensor
        如果batch_first为False，输出张量形状为[SeqLen, Batch, Embed]，否则为[Batch, SeqLen, Embed]
    """

    

    if batch_first is False:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    batch_size, q_seq_length, embed_dim = query.shape
    _, v_seq_length, _ = value.shape

    # -- 将输入投影到新的嵌入空间
    with torch.no_grad():
        q = torch.nn.functional.linear(query, q_proj_weight, bias=bias_q)
        k = torch.nn.functional.linear(key, k_proj_weight, bias=bias_k)
    v = v_proj(value)

    # -- 为多头注意力机制重塑形状
    q = q.view(batch_size, q_seq_length, num_heads, head_dim)
    k = k.view(batch_size, v_seq_length, num_heads, head_dim)
    v = v.view(batch_size, v_seq_length, num_heads, head_dim)

    q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # -- 在每个头上执行注意力计算
    with torch.no_grad():
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

        mask = torch.zeros_like(attn_logits).to(attn_logits)
        if key_padding_mask is not None:
            mask += _prepare_key_padding_mask(key_padding_mask, attn_mask, q)
        if attn_mask is not None:
            mask += _prepare_attn_mask(attn_mask, q)

        attn_logits = attn_logits + mask
        attention = torch.softmax(attn_logits, -1)

    # CP-LRP：不让相关性通过 softmax，值路径采用 epsilon-LRP；稳定项可配置
    y = rules.epsilon_lrp(torch.matmul, epsilon, attention.detach(), v)

    # -- 输出投影
    y = y.permute(0, 2, 1, 3)
    y = y.reshape(batch_size, q_seq_length, embed_dim)
    out = out_proj(y)

    if batch_first is False:
        out = out.transpose(0, 1)

    if need_weights and average_attn_weights:
        return out, attention.mean(dim=1)
    elif need_weights:
        return out, attention
    else:
        return out, None

@torch.fx.wrap
def multi_head_attention_attnlrp(query, key, value, batch_first, num_heads, head_dim, q_proj_weight, bias_q, k_proj_weight, bias_k, v_proj, out_proj,
                                 key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, epsilon: float = 1e-6):
    """
    AttnLRP 注意力：让相关性经由 softmax 与 Q/K 路径传播，(attention @ V) 使用 LRP 规则，
    将相关性分配至 attention 与 V，并继续回流至 Q/K。
    """

    if batch_first is False:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    batch_size, q_seq_length, embed_dim = query.shape
    _, v_seq_length, _ = value.shape

    # 投影到 Q/K/V
    with torch.no_grad():
        q = torch.nn.functional.linear(query, q_proj_weight, bias=bias_q)
        k = torch.nn.functional.linear(key, k_proj_weight, bias=bias_k)
    v = v_proj(value)

    # 重塑为多头
    q = q.view(batch_size, q_seq_length, num_heads, head_dim)
    k = k.view(batch_size, v_seq_length, num_heads, head_dim)
    v = v.view(batch_size, v_seq_length, num_heads, head_dim)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # logits 与 softmax（使用 LRP 版本 softmax，允许相关性沿 softmax 传播）
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    mask = torch.zeros_like(attn_logits).to(attn_logits)
    if key_padding_mask is not None:
        mask += _prepare_key_padding_mask(key_padding_mask, attn_mask, q)
    if attn_mask is not None:
        mask += _prepare_attn_mask(attn_mask, q)

    attn_logits = attn_logits + mask
    # 使用 LRP 版本 softmax（局部导入避免顶部编码问题）
    from .functional import softmax as lrp_softmax
    attention = lrp_softmax(attn_logits, -1)

    # 使用 epsilon-LRP 封装的 matmul，让相关性分配到 attention 与 v
    y = rules.epsilon_lrp(torch.matmul, epsilon, attention, v)

    # 输出投影
    y = y.permute(0, 2, 1, 3)
    y = y.reshape(batch_size, q_seq_length, embed_dim)
    out = out_proj(y)

    if batch_first is False:
        out = out.transpose(0, 1)

    if need_weights and average_attn_weights:
        return out, attention.mean(dim=1)
    elif need_weights:
        return out, attention
    else:
        return out, None
