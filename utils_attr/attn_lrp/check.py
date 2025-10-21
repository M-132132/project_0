# 导入必要的模块
import operator
import torch
import torch.nn as nn
from utils_attr.attn_lrp.functional import CONSERVATION_CHECK_FLAG

# 保守性检查类，用于验证LRP计算的保守性
class conservation_check(object):

    def __init__(self) -> None:
        """初始化保守性检查对象"""
        pass
            
    def __enter__(self):
        """进入上下文时启用保守性检查"""
        CONSERVATION_CHECK_FLAG[0] = True

    def __exit__(self, type, value, traceback):
        """退出上下文时禁用保守性检查"""
        CONSERVATION_CHECK_FLAG[0] = False

# 用于显示检查结果的符号字典
# true: 绿色对勾符号，false: 红色叉号符号，unknown: 黄色问号符号
SYMBOLS = {
    'true': '\033[0;32;40m \u2713 \033[0m',     # 绿色对勾
    'false': '\033[0;31;40m \u2717 \033[0m',    # 红色叉号
    'unknown': '\033[0;33;40m \u2047 \033[0m',  # 黄色问号
}

# 白名单：不影响LRP相关性传播的操作列表
# 这些操作不会改变相关性分数的总和
WHITELIST = [
    "transpose",      # 转置操作
    "view",           # 视图变换
    "unsqueeze",      # 增加维度
    "reshape",        # 重塑形状
    "permute",        # 维度重排
    "size",           # 获取尺寸
    "dim",            # 获取维度
    "expand",         # 扩展张量
    "to",             # 类型转换
    "argmax",         # 最大值索引

    operator.getitem, # 索引操作
    torch._assert,    # 断言
    operator.eq,      # 相等比较
    torch.cat,        # 张量拼接


]

# 黑名单：会影响LRP相关性传播的操作列表
# 这些操作可能会改变相关性分数的总和，需要特殊处理
BLACKLIST = [
    "sum",            # 求和操作
    "add",            # 加法操作
    torch.sum,        # PyTorch求和
    operator.add,     # 运算符加法
    
    operator.sub,     # 运算符减法

    "mul",            # 乘法操作
    operator.mul,     # 运算符乘法

    operator.floordiv, # 整除操作

    "mean",           # 均值操作
    torch.mean,       # PyTorch均值

    "matmul",         # 矩阵乘法
    torch.matmul,     # PyTorch矩阵乘法

    "softmax",        # Softmax激活
    torch.softmax,    # PyTorch Softmax

    "exp",            # 指数函数
]
