# utils/path_manager.py
import os
from pathlib import Path

try:
    from omegaconf import DictConfig, ListConfig
except ImportError:
    DictConfig = ()
    ListConfig = ()


class PathManager:
    def __init__(self):
        # 获取项目根目录（假设此文件在utils目录下）
        self.root_dir = Path(__file__).parent.parent.resolve()
    
    def get_root_path(self, *subdirs):
        """获取项目根目录路径"""
        return self.root_dir.joinpath(*subdirs)
        
    def get_config_path(self, *subdirs):
        """获取配置文件路径"""
        return self.root_dir.joinpath("configs", *subdirs)
    
    def get_utils_path(self, *subdirs):
        """获取utils相关路径"""
        return self.root_dir.joinpath("utils", *subdirs)
    
    def get_utils_attr_path(self, *subdirs):
        """获取归因工具相关路径"""
        return self.root_dir.joinpath("utils_attr", *subdirs)
    
    def get_dataset_path(self, *subdirs):
        """获取数据集相关路径"""
        return self.root_dir.joinpath("utils_datasets_traj", *subdirs)
    
    def get_models_path(self, *subdirs):
        """获取模型相关路径"""
        return self.root_dir.joinpath("models", *subdirs)
    
    def get_weights_path(self, *subdirs):
        """获取权重文件路径"""
        return self.root_dir.joinpath("weights", *subdirs)
    
    def get_ckpt_path(self, *subdirs):
        """获取检查点文件路径"""
        return self.root_dir.joinpath("TrajAttr_ckpt", *subdirs)
    
    def get_exp_res_path(self, *subdirs):
        """获取实验结果路径"""
        return self.root_dir.joinpath("exps_res", *subdirs)
    
    def get_exp_script_path(self, *subdirs):
        """获取实验脚本路径"""
        return self.root_dir.joinpath("exps_scripts", *subdirs)
    
    def get_logs_path(self, *subdirs):
        """获取日志文件路径"""
        return self.root_dir.joinpath("logs", *subdirs)
    
    def get_outputs_path(self, *subdirs):
        """获取输出文件路径"""
        return self.root_dir.joinpath("outputs", *subdirs)
    
    def resolve_path(self, path_str):
        """解析路径字符串，将相对路径转换为绝对路径"""
        if not isinstance(path_str, str):
            return path_str
            
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(path_str):
            return path_str
            
        # 处理特殊的相对路径
        if path_str.startswith('./cache') or path_str == './cache':
            return str(self.get_root_path('cache'))
        elif path_str.startswith('weights/'):



            return str(self.get_root_path(path_str))
        elif path_str.startswith('data_samples/'):
            return str(self.get_root_path(path_str))
        elif path_str.startswith('exps_res/'):
            return str(self.get_root_path(path_str))
        elif path_str.startswith('TrajAttr_ckpt/'):
            return str(self.get_root_path(path_str))
        elif path_str.startswith('./') or path_str.startswith('../'):
            # 其他相对路径都转换为基于项目根目录的路径
            clean_path = path_str.lstrip('./').replace('../', '')
            return str(self.get_root_path(clean_path))
        else:
            # 不以 / 开头的路径，假设是相对于项目根目录的
            return str(self.get_root_path(path_str))
    
    def resolve_config_paths(self, config_dict):

# """
    # 递归解析配置字典中的路径，将相对路径转换为绝对路径
    #
    # 参数:
    #     config_dict: 需要解析路径的配置字典或列表
    #
    # 处理逻辑:
    #     1. 如果是字典类型，检查每个键是否以'_path'或'_dir'结尾，或者是特定路径键
    #     2. 如果是字符串类型的值，直接解析路径
    #     3. 如果是列表或字典类型的值，递归处理
    #     4. 如果是列表类型，检查每个字符串项是否包含路径分隔符
        if isinstance(config_dict, (dict, DictConfig)):  # 处理字典类型
            for key in list(config_dict.keys()):  # 遍历字典键
                value = config_dict[key]  # 获取当前键的值
            # 检查键是否表示路径或目录
                if key.endswith('_path') or key.endswith('_dir') or key in ['cache_path', 'ckpt_path']:
                    if isinstance(value, str):  # 如果值是字符串，直接解析路径
                        config_dict[key] = self.resolve_path(value)
                    elif isinstance(value, (list, ListConfig)):  # 如果值是列表类型
                        for idx in range(len(value)):  # 遍历列表项
                            item = value[idx]
                            if isinstance(item, str):  # 如果列表项是字符串，解析路径
                                value[idx] = self.resolve_path(item)
                            elif isinstance(item, (dict, DictConfig, list, ListConfig)):  # 如果是复杂类型，递归处理
                                self.resolve_config_paths(item)
                        config_dict[key] = value
                    elif isinstance(value, (dict, DictConfig)):  # 如果值是字典类型，递归处理
                        self.resolve_config_paths(value)
                elif isinstance(value, (dict, DictConfig, list, ListConfig)):  # 处理其他复杂类型的值
                    self.resolve_config_paths(value)
        elif isinstance(config_dict, (list, ListConfig)):  # 处理列表类型
            for idx in range(len(config_dict)):  # 遍历列表项
                item = config_dict[idx]
            # 检查字符串项是否包含路径分隔符
                if isinstance(item, str) and ('/' in item or '\\' in item):
                    config_dict[idx] = self.resolve_path(item)  # 解析路径
                elif isinstance(item, (dict, DictConfig, list, ListConfig)):  # 如果是复杂类型，递归处理
                    self.resolve_config_paths(item)
# 创建全局路径管理器实例
path_manager = PathManager()
