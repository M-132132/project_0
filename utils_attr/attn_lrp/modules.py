import torch
import torch.nn as nn
import inspect
# 修改为相对导入：functional/special 使用本目录实现
from . import functional as lf
from . import special as ls
import torch.fx


###################
### LRP模块 ###
###################

class SoftmaxDT(nn.Softmax):

    """
    自定义的Softmax激活函数类，继承自nn.Softmax
    支持温度参数调节，可用于控制输出的平滑程度
    """
    def __init__(self, dim: int, dtype=None, temperature=1.0, inplace=False, **kwargs):

        """
        初始化函数
        参数:
            dim: 计算Softmax的维度
            dtype: 输出的数据类型
            temperature: 温度参数，用于调节Softmax的平滑程度
            inplace: 是否原地操作
            **kwargs: 其他可选参数
        """
        super().__init__(dim)  # 调用父类初始化方法
        self.inplace = inplace  # 设置是否原地操作
        self.dtype = dtype      # 设置输出数据类型
        self.temperature = temperature  # 设置温度参数

    def forward(self, inputs):
        """
        前向传播函数
        参数:
            inputs: 输入张量
        返回:
            应用Softmax激活后的结果
        """
        return lf.softmax(inputs, self.dim, self.dtype, self.temperature, self.inplace)


class LinearEpsilon(nn.Linear):

    """
    继承自nn.Linear的自定义线性层，添加了epsilon参数用于数值稳定性处理。
    参数:
        in_features (int): 输入特征的数量
        out_features (int): 输出特征的数量
        bias (bool, optional): 是否添加偏置项，默认为True
        device: 指定张量存储的设备，默认为None
        dtype: 指定张量的数据类型，默认为None
        epsilon (float, optional): 用于数值稳定性的小值，默认为1e-6
        **kwargs: 其他传递给父类nn.Linear的参数
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, epsilon=1e-6, **kwargs):
        """
        初始化函数
        参数:
            in_features (int): 输入特征的维度
            out_features (int): 输出特征的维度
            bias (bool): 是否使用偏置项，默认为True
            device: 指定运行设备，默认为None
            dtype: 指定数据类型，默认为None
            epsilon (float): 用于数值稳定性处理的小量，默认为1e-6
            **kwargs: 其他可选参数
        """
        # 调用父类nn.Linear的初始化方法
        super().__init__(in_features, out_features, bias, device, dtype)
        # 初始化epsilon参数，用于数值稳定性处理
        self.epsilon = epsilon

    def forward(self, inputs):
        """
        前向传播方法，使用自定义的linear_epsilon函数计算输出。
        参数:
            inputs: 输入张量
        返回:
            应用线性变换并添加epsilon数值稳定性处理后的输出
        """
        # 调用自定义的linear_epsilon函数进行计算
        return lf.linear_epsilon(inputs, self.weight, self.bias, self.epsilon)
    

class RMSNormIdentity(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        初始化RMSNormIdentity模块
        参数:
            hidden_size (int): 隐藏层的大小
            eps (float, optional): 用于数值稳定性的小常数，默认为1e-6
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化可训练的权重参数，默认值为1
        self.variance_epsilon = eps  # 设置方差计算中的小常数，防止除以0

    def forward(self, hidden_states):

        """
        前向传播函数

        参数:
            hidden_states (torch.Tensor): 输入的张量

        返回:
            torch.Tensor: 经过RMS归一化处理后的张量
        """
        return lf.rms_norm_identity(hidden_states, self.weight, self.variance_epsilon)  # 调用自定义的RMS归一化函数
    

class LayerNormEpsilon(nn.LayerNorm):

    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)

    def forward(self, x):
        return lf.layer_norm(x, self.weight, self.bias, self.eps)
    

##################################
### 多头注意力模块 ###
##################################

class LinearInProjection(nn.Module):
    """
    自定义nn.Linear模块，便于为其附加不同的规则。
    """
    def __init__(self, weight, bias):
        super().__init__()

        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
class LinearOutProjection(nn.Module):
    """
    自定义nn.Linear模块，便于为其附加不同的规则。
    """
    def __init__(self, weight, bias):
        super().__init__()

        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class MultiheadAttention_CP(nn.Module):
    """
    实现注意力机制的CP-LRP（保守传播-LRP）规则，即我们不让相关性通过softmax流动，而只通过值路径。
    这种方法*仅在视觉变换器中效果良好*，因为在这里高级的AttnLRP规则（确实使用softmax）与CP-LRP规则具有相似的性能。
    AttnLRP的问题在于使用softmax会引入梯度破碎，这需要应用z-plus LRP规则。
    这使得AttnLRP效率稍低，并且根据我们有限的实验，在视觉变换器中小幅性能提升不值得。
    然而，在大型语言模型中，在softmax上应用AttnLRP比CP-LRP要好得多，并且不需要效率较低的z-plus规则。
    因此，我们为注意力选择更高效的CP-LRP，并为ViT的其他部分使用AttnLRP。

    请参考论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的A.2.3节
    '解决视觉变换器中的噪声'。
    """
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()

        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj = LinearInProjection(None, None)
        self.out_proj = LinearOutProjection(None, None)

        self.embed_dim = None
        self.num_heads = None
        self.head_dim = None
        self.batch_first = None

        self.bias_q = None
        self.bias_k = None
        self.epsilon = float(epsilon)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):

        assert is_causal == False # 暂不支持

        cp = ls.multi_head_attention_cp(query, key, value, self.batch_first, self.num_heads, self.head_dim,
                                        self.q_proj_weight, self.bias_q, self.k_proj_weight, self.bias_k, self.v_proj,
                                        self.out_proj, key_padding_mask, need_weights, attn_mask, average_attn_weights,
                                        epsilon=self.epsilon)
        return cp


class MultiheadAttention_AttnLRP(nn.Module):
    """
    完整 AttnLRP 注意力：允许相关性沿 softmax/Q/K 路径传播。
    """
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()

        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj = LinearInProjection(None, None)
        self.out_proj = LinearOutProjection(None, None)

        self.embed_dim = None
        self.num_heads = None
        self.head_dim = None
        self.batch_first = None

        self.bias_q = None
        self.bias_k = None
        self.epsilon = float(epsilon)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        assert is_causal == False

        attn = ls.multi_head_attention_attnlrp(query, key, value, self.batch_first, self.num_heads, self.head_dim,
                                               self.q_proj_weight, self.bias_q, self.k_proj_weight, self.bias_k,
                                               self.v_proj, self.out_proj, key_padding_mask, need_weights,
                                               attn_mask, average_attn_weights, epsilon=self.epsilon)
        return attn
    
    
##########################
### 初始化模块 ###
##########################

def copy_parameters_and_buffers_(original, replacement):
    """
    复制原始模块的参数和缓冲区。
    """

    for name, param in original.named_parameters():
        replacement.register_parameter(name, param)

    for name, buffer in original.named_buffers():
        replacement.register_buffer(name, buffer)


def initialize_generic(original, replacement):
    """
    使用正确的参数初始化替换模块。
    """
    
    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    replacement = replacement(**kwargs)
    copy_parameters_and_buffers_(original, replacement)

    return replacement


def initialize_bias(original, replacement):
    """
    使用bias参数正确初始化LinearEpsilon模块。
    """

    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    kwargs["bias"] = True if original.bias is not None else False

    replacement = replacement(**kwargs)
    copy_parameters_and_buffers_(original, replacement)

    return replacement


def initialize_MHA(original, replacement):
    """
    初始化MultiheadAttention_CP模块。
    """
    
    replacement = replacement()
    
    if not original._qkv_same_embed_dim:
        replacement.q_proj_weight = original.q_proj_weight
        replacement.k_proj_weight = original.k_proj_weight
        replacement.v_proj.weight = original.v_proj.weight
    else:
        replacement.q_proj_weight = original.in_proj_weight[:original.embed_dim]
        replacement.k_proj_weight = original.in_proj_weight[original.embed_dim:original.embed_dim*2]
        replacement.v_proj.weight = original.in_proj_weight[original.embed_dim*2:original.embed_dim*3]

    if original.in_proj_bias is not None:
        replacement.bias_q = original.in_proj_bias[:original.embed_dim]
        replacement.bias_k = original.in_proj_bias[original.embed_dim:original.embed_dim*2]
        replacement.v_proj.bias = original.in_proj_bias[original.embed_dim*2:original.embed_dim*3]
        
    if original.bias_k is not None:
        raise NotImplementedError("暂不支持add_bias_kv=True。")
    
    replacement.out_proj.weight = original.out_proj.weight
    replacement.out_proj.bias = original.out_proj.bias

    replacement.embed_dim = original.embed_dim
    replacement.num_heads = original.num_heads
    replacement.head_dim = original.head_dim
    replacement.batch_first = original.batch_first

    return replacement


INIT_MODULE_MAPPING = {
    SoftmaxDT: initialize_generic,
    LinearEpsilon: initialize_bias,
    RMSNormIdentity: initialize_generic,
    LayerNormEpsilon: initialize_bias,
    MultiheadAttention_CP: initialize_MHA,
    MultiheadAttention_AttnLRP: initialize_MHA,
}
