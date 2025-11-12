import torch
from torch.autograd import Function
import torch.nn as nn
# 修改为相对导入：使用本目录下的 functional 实现
from .functional import _stabilize, conservation_check_wrap
from torch.func import jvp, vjp
import torch.fx

class WrapModule(nn.Module):
    """
    用于将规则包装在模块周围的基类。这个类不意在直接使用，而是由特定规则的子类继承。
    然后用于用规则包装的模块替换原始模块。
    """

    def __init__(self, module):
        super(WrapModule, self).__init__()
        self.module = module


class IdentityRule(WrapModule):
    """
    根据论文中的等式9的恒等规则，将100%的相关性分配给输入：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    参数:
    -----------

    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分，并且有单一输入和输出
    input: torch.Tensor
        输入张量
    """

    def forward(self, input):

        return identity_fn.apply(self.module, input)
    

def identity(fn, input):
    """
    根据论文中的等式9的恒等规则，将100%的相关性分配给输入：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    参数:
    -----------

    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分，并且有单一输入和输出
    input: torch.Tensor
        输入张量
    """
    return identity_fn.apply(fn, input)


class identity_fn(Function):
    """
    根据论文中的等式9的恒等规则，将100%的相关性分配给输入：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    参数:
    -----------

    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分，并且有单一输入和输出
    input: torch.Tensor
        输入张量
    """

    @staticmethod
    def forward(ctx, fn, input):

        output = fn(input)
        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return (None,) + out_relevance
    

class StopRelevanceRule(WrapModule):
    """
    在输入处停止相关性流动。等同于PyTorch中的.detach()。

    参数:
    -----------

    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分，并且有单一输入和输出
    input: torch.Tensor
        输入张量
    """

    def forward(self, input):

        return stop_relevance_fn.apply(self.module, input)
    

class stop_relevance_fn(Function):
    """
    在输入处停止相关性流动。等同于PyTorch中的.detach()。

    参数:
    -----------

    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分，并且有单一输入和输出
    input: torch.Tensor
        输入张量
    """

    @staticmethod
    def forward(ctx, fn, input):

        output = fn(input)
        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return (None, None)


class EpsilonRule(WrapModule):
    """
    梯度 X 输入（带有偏置的泰勒分解或线性层的标准Epsilon-LRP规则），根据论文的等式4-5和8：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    如果其中一个输入是常数或不需要梯度，则不会向其分配相关性。

    参数:
    -----------
    module: nn.Module
        要被包装的模块
    epsilon: float
        用于稳定input_x_gradient规则中分母的小值

    """

    def __init__(self, module, epsilon=1e-8):
        
        super(EpsilonRule, self).__init__(module)
        self.epsilon = epsilon

    def forward(self, *inputs):

        return epsilon_lrp_fn.apply(self.module, self.epsilon, *inputs)

@torch.fx.wrap
def epsilon_lrp(fn, epsilon, *inputs):
    """
    梯度 X 输入（带有偏置的泰勒分解或线性层的标准Epsilon-LRP规则），根据论文的等式4-5和8：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    如果其中一个输入是常数或不需要梯度，则不会向其分配相关性。

    参数:
    -----------
    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分
    epsilon: float
        用于稳定分母的小值
    *inputs: 至少一个torch.Tensor
        函数的输入张量
    """
    return epsilon_lrp_fn.apply(fn, epsilon, *inputs)


class epsilon_lrp_fn(Function):
    """
    梯度 X 输入（带有偏置的泰勒分解或线性层的标准Epsilon-LRP规则），根据论文的等式4-5和8：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    如果其中一个输入是常数或不需要梯度，则不会向其分配相关性。

    参数:
    -----------
    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分
    epsilon: float
        用于稳定分母的小值
    *inputs: 至少一个torch.Tensor
        函数的输入张量
    """

    @staticmethod
    def forward(ctx, fn, epsilon, *inputs):

        # 为需要梯度的输入创建布尔掩码
        #TODO: 使用ctx.needs_input_grad而不是requires_grad
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            # 没有梯度需要计算或使用了梯度检查点
            return fn(*inputs)
        
        # 分离输入以避免在同一输入用作多个参数时覆盖梯度（如在自注意力中）
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.requires_grads = epsilon, requires_grads
        # 只保存需要梯度的输入
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, outputs)
        
        return outputs.detach()
        
    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, outputs = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        relevance_norm = out_relevance[0] / _stabilize(outputs, ctx.epsilon, inplace=False)

        # 计算向量-雅可比积
        grads = torch.autograd.grad(outputs, inputs, relevance_norm)

        # 在requires_grad索引处返回相关性，否则返回None
        relevance = iter([grads[i].mul_(inputs[i]) for i in range(len(inputs))])
        return (None, None) + tuple(next(relevance) if req_grad else None for req_grad in ctx.requires_grads)

        
    

class UniformEpsilonRule(WrapModule):
    """
    输入_x_梯度规则和均匀规则的顺序应用，根据均匀规则将相关性均匀分配给所有输入，
    如论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的第3.3.2节“处理矩阵乘法”中所讨论。

    如果其中一个输入是常数或不需要梯度，则不会向其分配相关性。

    参数:
    -----------
    module: nn.Module
        要被包装的模块
    epsilon: float
        用于稳定输入_x_梯度规则中分母的小值

    """

    def __init__(self, module, epsilon=1e-6):
        
        super(UniformEpsilonRule, self).__init__(module)
        self.epsilon = epsilon

    def forward(self, *inputs):

        return uniform_epsilon_lrp_fn.apply(self.module, self.epsilon, *inputs)
    

class uniform_epsilon_lrp_fn(epsilon_lrp_fn):
    """
    输入_x_梯度规则和均匀规则的顺序应用，根据均匀规则将相关性均匀分配给所有输入，
    如论文'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'的第3.3.2节“处理矩阵乘法”中所讨论。

    如果其中一个输入是常数或不需要梯度，则不会向其分配相关性。

    参数:
    -----------
    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分
    epsilon: float
        用于稳定分母的小值
    *inputs: 至少一个torch.Tensor
        函数的输入张量
    """
        
    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):
    
        inputs, outputs = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        relevance_norm = out_relevance[0] / _stabilize(outputs, ctx.epsilon, inplace=False)
        relevance_norm = relevance_norm / len(inputs)

        # 计算向量-雅可比积
        grads = torch.autograd.grad(outputs, inputs, relevance_norm)

        # 在requires_grad索引处返回相关性，否则返回None
        return (None, None) + tuple(grads[i].mul_(inputs[i]) if ctx.requires_grads[i] else None for i in range(len(ctx.requires_grads)))



class TaylorDecompositionRule(WrapModule):
    """
    适用于任何可微分函数的广义泰勒分解（带有或不带有偏置），根据论文的等式4-5
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    注意：所有输入必须是张量并且都将接收相关性！

    参数:
    -----------
    module: nn.Module
        要被包装的模块
    ref: torch.Tensor的可迭代对象
        雅可比计算的参考点
    bias: bool
        是否在相关性计算中包含偏置项
    distribute_bias: callable
        用于将偏置相关性分配给输入张量的函数，仅在bias=True时使用
    """

    def __init__(self, module, ref=0, bias=False, distribute_bias=None):
        super(TaylorDecompositionRule, self).__init__(module)
        self.ref = ref
        self.bias = bias
        self.distribute_bias = distribute_bias

    def forward(self, *inputs):

        return taylor_decomposition_fn.apply(self.module, self.ref, self.bias, self.distribute_bias, *inputs)



class taylor_decomposition_fn(Function):
    """
    适用于任何可微分函数的广义泰勒分解（带有或不带有偏置），根据论文的等式4-5：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    所有输入必须是张量并且都将接收相关性。如果您希望将某个张量从相关性计算中排除，您必须相应地包装该函数。

    参数:
    -----------
    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分
    ref: torch.Tensor的可迭代对象
        雅可比计算的参考点
    bias: bool
        是否在相关性计算中包含偏置项
    distribute_bias: callable
        用于将偏置相关性分配给输入张量的函数，仅在bias=True时使用
    *inputs: 所有torch.Tensor
        函数的输入张量
    """

    @staticmethod
    def forward(ctx, fn, ref, bias, distribute_bias, *inputs):
        
        output = fn(*inputs)
        ctx.save_for_backward(*inputs)
        ctx.fn, ctx.ref, ctx.bias = fn, ref, bias
        ctx.distribute_bias = distribute_bias

        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs = ctx.saved_tensors

        if not ctx.bias:
            # 在参考点计算雅可比并从右侧与输入相乘
            # 这样就省略了偏置项
            _, Jvs = jvp(ctx.fn, ctx.ref, inputs)
            output = Jvs
        
        normed_relevance = out_relevance[0] / _stabilize(output, inplace=True)

        # 在参考点计算雅可比并从左侧与R/output相乘
        _, vjpfunc = vjp(ctx.fn, *ctx.ref)
        grads = vjpfunc(normed_relevance)
        
        relevances = tuple(grads[i].mul_(inputs[i]) for i in range(len(inputs)))

        if ctx.bias and callable(ctx.distribute_bias):
            relevances = ctx.distribute_bias(inputs, relevances)

        # vJ与参考点相乘
        return (None, None, None) + relevances
    

class UniformRule(WrapModule):
    """
    根据论文等式7中的均匀规则，将相关性均匀分配给所有输入：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    参数:
    -----------
    module: nn.Module
        要被包装的模块
    """

    def forward(self, *inputs):

        return uniform_rule_fn.apply(self.module, *inputs)


class uniform_rule_fn(Function):
    """
    根据论文等式7中的均匀规则，将相关性均匀分配给所有输入：
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    参数:
    -----------
    fn: callable
        要用输入调用的函数，必须在PyTorch中可微分
    *inputs: 所有torch.Tensor
        函数的输入张量
    """

    @staticmethod
    def forward(ctx, fn, *inputs):

        output = fn(*inputs)
        ctx.save_for_backward(*inputs)

        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs = ctx.saved_tensors
        n = len(inputs)
        return (None,) + tuple(out_relevance[0] / n for _ in range(n))
