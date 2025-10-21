import torch
import torch.fx
from torch.autograd import Function
import torch.nn.functional as F

####################
###   健全性检查   ###
####################

CONSERVATION_CHECK_FLAG = [False]

def conservation_check_wrap(func):
    #TODO: add2_fn中的bug
    """
    装饰器，用于启用或禁用LRP操作的健全性检查，即测试LRP守恒属性是否适用于除偏置项之外的所有操作。
    如果启用健全性检查，相关性将均匀分布到输入张量，否则相关性将按函数计算的方式返回。
    此检查有助于验证模型中使用的所有操作是否都与LRP兼容。
    """
    def wrapped(ctx, *out_relevance):

        inp_relevance = func(ctx, *out_relevance)

        if CONSERVATION_CHECK_FLAG[0]:

            out_rel_sum = sum(r.float().sum() for r in out_relevance if r is not None)
            inp_elements = sum(r.numel() for r in inp_relevance if r is not None)
            inp_rel_mean = out_rel_sum/inp_elements

            if torch.isnan(inp_rel_mean).any():
                raise ValueError(f"NaN at {func}")
            
            inp_relevance = tuple(torch.full(r.shape, inp_rel_mean).to(r.device) if r is not None else None for r in inp_relevance)


        return inp_relevance
        
    return wrapped

#####################
###   LRP 函数     ###
#####################

@torch.fx.wrap
def add2(input_a, input_b, inplace=False, epsilon=1e-8):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的公式8，
    用于两个张量按元素相加（沿所有维度）的标准Epsilon-LRP规则

    参数:
    -----------
    input_a: torch.Tensor
        第一个输入张量
    input_b: torch.Tensor
        第二个输入张量
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    epsilon: float
        用于稳定分母的小值
    """
    return add2_tensors_fn.apply(input_a, input_b, inplace, epsilon)

@torch.fx.wrap
def softmax(input, dim, dtype=None, temperature=1.0, inplace=False):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.1，
    在x处使用深度泰勒分解（带偏置）计算相关性

    参数:
    -----------
    input: torch.Tensor
        输入张量
    dim: int
        应用softmax函数的维度
    dtype: torch.dtype
        在应用softmax函数之前将输入转换为此数据类型
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    """
    return softmax_fn.apply(input, dim, dtype, temperature, inplace)

@torch.fx.wrap
def linear_epsilon(input, weight, bias=None, epsilon=1e-6):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的公式8
    或论文'On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation'的公式16，
    用于nn.functional.linear的标准Epsilon-LRP规则

    参数:
    -----------
    input: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    bias: torch.Tensor
        偏置张量
    epsilon: float
        用于稳定分母的小值
    """
    return linear_epsilon_fn.apply(input, weight, bias, epsilon)

@torch.fx.wrap
def matmul(input_a, input_b, inplace=False, epsilon=1e-8):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.3，
    通过顺序应用epsilon-LRP规则和uniform-LRP规则来计算相关性

    参数:
    -----------
    input_a: torch.Tensor
        第一个输入张量
    input_b: torch.Tensor
        第二个输入张量
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    epsilon: float
        用于稳定分母的小值
    """
    return matmul_fn.apply(input_a, input_b, inplace, epsilon)

@torch.fx.wrap
def rms_norm_identity(hidden_states, weight, variance_epsilon):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4和公式9，
    计算LlamaRMSNorm层的相关性

    由于我们也对weight * hidden_states.to(input_dtype)应用恒等规则，我们可以对整个层应用恒等规则，
    即将相关性100%分配给输入。

    参数:
    -----------
    hidden_states: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    variance_epsilon: float
        用于稳定分母的小值
    """
    return rms_norm_identity_fn.apply(hidden_states, weight, variance_epsilon)

@torch.fx.wrap
def mul2(input_a, input_b, inplace=False):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.2，
    用于两个张量按元素相乘（沿所有维度）的均匀LRP规则

    如果其中一个输入是常数或不需要梯度，相关性将100%分配给另一个输入。

    参数:
    -----------
    input_a: torch.Tensor
        第一个输入张量
    input_b: torch.Tensor
        第二个输入张量
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    """
    return mul2_fn.apply(input_a, input_b, inplace)

@torch.fx.wrap
def mean(x, dim, keep_dim, epsilon=1e-6):
    """
    用于均值操作的Epsilon LRP规则。

    参数:
    -----------
    x: torch.Tensor
        输入张量
    dim: int
        应用均值函数的维度
    keep_dim: bool
        应用均值函数后是否保持维度
    epsilon: float
        用于稳定分母的小值
    """
    
    return mean_fn.apply(x, dim, keep_dim, epsilon)

@torch.fx.wrap
def layer_norm(hidden_states, weight, bias, variance_epsilon):
    """
    标准nn.LayerNorm操作的恒等和epsilon规则混合：
    对按元素操作(y * weight)使用恒等规则，因为单输入单输出。根据论文
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4，
    对1/std使用恒等规则。(x - mean)是线性操作，所以我们对其应用epsilon规则。

    为了实现这一点，我们使用一个技巧：我们对整个层进行微分，同时将std操作从图中分离。
    这等同于上述讨论的所有规则！这比用纯lxt实现所有内容稍快。
    参见_layer_norm_slower获取纯lxt实现。

    参数:
    -----------
    hidden_states: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    variance_epsilon: float
        用于稳定分母的小值
    """

    return layer_norm_grad_fn.apply(hidden_states, weight, bias, variance_epsilon)

@torch.fx.wrap
def _layer_norm_slower(hidden_states, weight, bias, variance_epsilon):
    """
    标准nn.LayerNorm操作的恒等和epsilon规则混合：
    对按元素操作(y * weight)使用恒等规则，因为单输入单输出。根据论文
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4，
    对1/std使用恒等规则。(x - mean)是线性操作，所以我们对其应用epsilon规则。

    此实现比layer_norm函数慢，因为它使用纯lxt实现而不是使用torch.autograd。

    参数:
    -----------
    hidden_states: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    variance_epsilon: float
        用于稳定分母的小值
    """

    
    x_mean = mean(hidden_states, -1, keep_dim=True)

    # 由于我们分离了std，所以相关性不会通过std流动！
    var = ((hidden_states - x_mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + variance_epsilon).sqrt().detach()
    
    y = add2(hidden_states, mul2(x_mean, -1))
    # 如果第二个输入不需要梯度，mul2就是恒等的
    y = mul2(y, 1/std)
    y = mul2(y, weight)
    y = add2(y, bias)

    return y


@torch.fx.wrap
def normalize(input, p= 2.0, dim= 1, eps= 1e-12, out=None):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4，
    对torch.nn.functional.normalize操作应用恒等规则


    参数:
    -----------
    input: torch.Tensor
        输入张量
    p: float
        范数计算的幂次
    dim: int
        应用归一化的维度
    eps: float
        用于稳定分母的小值
    """

    assert out is None, "不支持out参数"
    
    return normalize_identity_fn.apply(input, p, dim, eps)

###############################
###     自动微分实现        ###
###############################

def _stabilize(input, epsilon=1e-6, inplace=False):
    """
    通过添加小值来稳定输入
    """
    if inplace:
        return input.add_(epsilon)
    else:
        return input + epsilon
    

class softmax_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.1，
    在x处使用深度泰勒分解（带偏置）计算相关性

    参数:
    -----------
    input: torch.Tensor
        输入张量
    dim: int
        应用softmax函数的维度
    dtype: torch.dtype
        在应用softmax函数之前将输入转换为此数据类型
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    """

    @staticmethod
    def forward(ctx, inputs, dim, dtype=None, temperature=1.0, inplace=False):

        if dtype is not None:
            inputs = inputs.to(dtype)

        inputs = inputs / temperature
    
        outputs = F.softmax(inputs, dim=dim, dtype=dtype)

        ctx.save_for_backward(inputs, outputs)
        ctx.inplace = inplace

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, output = ctx.saved_tensors

        # 为了数值稳定性，将所有-inf（来自注意力掩码）替换为0
        inputs = torch.where(torch.isneginf(inputs), torch.tensor(0).to(inputs), inputs)

        if ctx.inplace:
            relevance = (out_relevance[0].sub_(output.mul_(out_relevance[0].sum(-1, keepdim=True)))).mul_(inputs)
        else:
            relevance = inputs * (out_relevance[0] - output * out_relevance[0].sum(-1, keepdim=True))
        
        return (relevance, None, None, None, None)


class linear_epsilon_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的公式8
    或论文'On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation'的公式16，
    用于nn.functional.linear的标准Epsilon-LRP规则

    参数:
    -----------
    input: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    bias: torch.Tensor
        偏置张量
    epsilon: float
        用于稳定分母的小值
    """

    @staticmethod
    def forward(ctx, inputs, weight, bias=None, epsilon=1e-6):
        
        outputs = F.linear(inputs, weight, bias)
        ctx.save_for_backward(inputs, weight, outputs)
        ctx.epsilon = epsilon
    
        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, weight, outputs = ctx.saved_tensors
        epsilon = ctx.epsilon

        relevance_norm = out_relevance[0] / _stabilize(outputs, epsilon)

        relevance = torch.matmul(relevance_norm, weight).mul_(inputs)
        
        return (relevance, None, None, None)


class matmul_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.3，
    通过顺序应用epsilon-LRP规则和uniform-LRP规则来计算相关性

    参数:
    -----------
    input_a: torch.Tensor
        第一个输入张量
    input_b: torch.Tensor
        第二个输入张量
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    epsilon: float
        用于稳定分母的小值
    """
    
    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False, epsilon=1e-6):
        
        outputs = torch.matmul(input_a, input_b)
        ctx.save_for_backward(input_a, input_b, outputs)
        ctx.inplace, ctx.epsilon = inplace, epsilon

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        input_a, input_b, outputs = ctx.saved_tensors
        inplace, epsilon = ctx.inplace, ctx.epsilon

        if inplace:
            relevance_norm = out_relevance[0].div_(_stabilize(outputs.mul_(2), epsilon, inplace))
        else:
            relevance_norm = out_relevance[0] / _stabilize(outputs * 2, epsilon, inplace)

        relevance_a = torch.matmul(relevance_norm, input_b.transpose(-1, -2)).mul_(input_a)
        relevance_b = torch.matmul(input_a.transpose(-1, -2), relevance_norm).mul_(input_b)
        
        return (relevance_a, relevance_b, None, None)



class add2_tensors_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的公式8，
    用于两个张量按元素相加（沿所有维度）的标准Epsilon-LRP规则

    参数:
    -----------
    input_a: torch.Tensor
        第一个输入张量
    input_b: torch.Tensor
        第二个输入张量
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    epsilon: float
        用于稳定分母的小值
    """
    
    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False, epsilon=1e-6):
    
        outputs = input_a + input_b
        if any([inp.requires_grad for inp in (input_a, input_b)]):
            ctx.save_for_backward(input_a, input_b)
            ctx.epsilon, ctx.inplace = epsilon, inplace

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        #TODO: 用requires grad相关内容替换守恒检查

        input_a, input_b = ctx.saved_tensors

        if ctx.inplace:
            relevance_norm = out_relevance[0].div_(_stabilize(input_a + input_b, epsilon=ctx.epsilon, inplace=True))

            relevance_a = relevance_norm * input_a
            relevance_b = relevance_norm.mul_(input_b)

        else:
            relevance_norm = out_relevance[0] / _stabilize(input_a + input_b, epsilon=ctx.epsilon, inplace=True)

            relevance_a = relevance_norm * input_a
            relevance_b = relevance_norm * input_b

        return (relevance_a, relevance_b, None, None)



class rms_norm_identity_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4和公式9，
    计算LlamaRMSNorm层的相关性

    由于我们也对weight * hidden_states.to(input_dtype)应用恒等规则，我们可以对整个层应用恒等规则，
    即将相关性100%分配给输入。

    参数:
    -----------
    hidden_states: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    variance_epsilon: float
        用于稳定分母的小值
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

        return weight * hidden_states.to(input_dtype)

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return out_relevance + (None, None)


class mul2_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.2，
    用于两个张量按元素相乘（沿所有维度）的均匀LRP规则

    如果其中一个输入是常数或不需要梯度，相关性将100%分配给另一个输入。

    参数:
    -----------
    input_a: torch.Tensor
        第一个输入张量
    input_b: torch.Tensor
        第二个输入张量
    inplace: bool
        在反向传播过程中是否就地执行操作，将覆盖输出处的相关性
    """


    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False):

        ctx.requires_grads = [i for i, inp in enumerate((input_a, input_b)) if isinstance(inp, torch.Tensor) and inp.requires_grad]
        ctx.inplace = inplace

        return input_a * input_b

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        n_required = len(ctx.requires_grads)

        if ctx.inplace:
            out_relevance = out_relevance[0].div_(n_required)
        else:
            out_relevance = out_relevance[0] / n_required

        # 只在requires_grad索引处返回相关性，否则返回None
        return tuple(out_relevance if i in ctx.requires_grads else None for i in range(2)) + (None,)


class mean_fn(Function):
    """
    用于均值操作的Epsilon LRP规则。

    参数:
    -----------
    x: torch.Tensor
        输入张量
    dim: int
        应用均值函数的维度
    keep_dim: bool
        应用均值函数后是否保持维度
    epsilon: float
        用于稳定分母的小值
    """

    @staticmethod
    def forward(ctx, x, dim, keepdim, epsilon=1e-6):

        y = x.mean(dim, keepdim)
    
        ctx.save_for_backward(x)
        ctx.epsilon, ctx.dim, ctx.keepdim = epsilon, dim, keepdim

        return y

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        x, = ctx.saved_tensors

        x_sum = x.sum(ctx.dim, keepdim=True)

        if ctx.keepdim is False:
            out_relevance = out_relevance[0].unsqueeze(ctx.dim)
        else:
            out_relevance = out_relevance[0]

        relevance = x * out_relevance / _stabilize(x_sum, ctx.epsilon, inplace=True)

        if ctx.keepdim is False:
            relevance = relevance.squeeze(ctx.dim)

        return (relevance, None, None, None)
    

class layer_norm_grad_fn(Function):
    """
    标准nn.LayerNorm操作的恒等和epsilon规则混合：
    对按元素操作(y * weight)使用恒等规则，因为单输入单输出。根据论文
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4，
    对1/std使用恒等规则。(x - mean)是线性操作，所以我们对其应用epsilon规则。

    为了实现这一点，我们使用一个技巧：我们对整个层进行微分，同时将std操作从图中分离。
    这等同于上述讨论的所有规则！这比用纯lxt实现所有内容稍快。

    参数:
    -----------
    hidden_states: torch.Tensor
        输入张量
    weight: torch.Tensor
        权重张量
    variance_epsilon: float
        用于稳定分母的小值
    """

    @staticmethod
    def forward(ctx, x, weight, bias, variance_epsilon, epsilon=1e-6):

        with torch.enable_grad():

            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            std = (var + variance_epsilon).sqrt()
            y = (x - mean) / std.detach() # 分离std操作将其从计算图中移除，即对x/std应用恒等规则
            if weight is not None:
                y *= weight
            if bias is not None:
                y += bias

            ctx.save_for_backward(x, y)
            ctx.epsilon = epsilon

        return y.detach()

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        x, y = ctx.saved_tensors

        relevance_norm = out_relevance[0] / _stabilize(y, ctx.epsilon, False)

        grads, = torch.autograd.grad(y, x, relevance_norm)

        return (grads*x, None, None, None, None)


class normalize_identity_fn(Function):
    """
    根据论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的命题3.4，
    对torch.nn.functional.normalize操作应用恒等规则


    参数:
    -----------
    input: torch.Tensor
        输入张量
    p: float
        范数计算的幂次
    dim: int
        应用归一化的维度
    eps: float
        用于稳定分母的小值
    """

    @staticmethod
    def forward(ctx, input, p, dim, eps):

        return F.normalize(input, p=p, dim=dim, eps=eps)

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return out_relevance + (None, None, None)