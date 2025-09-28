# 轨迹预测归因计算框架 (TrajAttr)

该框架为轨迹预测模型提供统一的归因计算接口，支持多种归因方法和模型类型。采用模型适配器架构，自动适配不同模型的输入输出格式。

## 功能特性

- **多模型支持**: 通过适配器模式支持AutoBot, Wayformer, MTR, SMART等轨迹预测模型
- **多归因方法**: 自定义Dirichlet方法 + Captum库集成（15+种方法）
- **智能适配**: 自动识别模型类型并适配输入输出格式
- **灵活距离度量**: ADE, FDE, L1, L2等多种距离计算
- **模块化设计**: 可扩展的模块化架构，易于添加新模型和方法

## 架构设计

### 新的模块化架构

```
utils_attr/traj_attr/
├── __init__.py                 # 框架入口
├── README.md                   # 使用说明
├── base/                       # 基础框架
│   ├── traj_attr_base.py      # 基础归因类（统一接口）
│   └── distance_metrics.py   # 距离度量函数
├── adapters/                   # 模型适配器（新增）
│   ├── __init__.py
│   └── model_adapters.py      # 统一的模型适配器系统
├── methods/                    # 归因方法
│   ├── dirichlet_attr.py      # Dirichlet方法
│   ├── guided_ig_attr.py      # Guided-IG方法（新增）
│   └── captum_attr.py         # Captum方法集成
├── models/                     # 已废弃！请使用adapters/
│   └── __init__.py            # 标记为废弃的旧适配器
└── examples/                   # 使用示例（新增）
    ├── __init__.py
    └── usage_example.py        # 完整使用示例
```

### 核心组件

1. **TrajAttrBase**: 统一的归因计算接口和保存管理
2. **ModelAdapterFactory**: 自动创建适配不同模型的适配器
3. **BaseModelAdapter**: 模型适配器基类，定义统一接口
4. **具体适配器**: AutoBotAdapter, WayformerAdapter, MTRAdapter等
5. **归因方法**: 专注于计算，不处理保存（单一职责原则）

### 关键设计原则

- **统一配置传递**: DictConfig 直接传递到归因方法，无需重构
- **统一保存管理**: 所有保存逻辑在 TrajAttrBase 中统一处理  
- **单一职责**: 归因方法只负责计算，适配器只负责输入输出适配
- **模块化配置**: 每个归因方法有独立的配置文件
- **向后兼容**: 支持传统 dict 配置格式

## 快速开始

### 1. 统一接口使用（推荐）

```python
from utils_attr.traj_attr.base.traj_attr_base import TrajAttrBase
from omegaconf import DictConfig

# 方式1：使用 Hydra DictConfig（推荐）
# 配置文件会自动加载 guided_ig_config, dirichlet_config, captum_config
attr_calculator = TrajAttrBase(model, hydra_config)

# 方式2：手动构建配置（向后兼容）
config = {
    'model_name': 'autobot',
    'attribution': {'methods': ['GuidedIG', 'Dirichlet']},
    'guided_ig_config': {'steps': 50, 'fraction': 0.1, 'anchors': 10},
    'dirichlet_config': {'alpha': 0.1, 'n_paths': 9, 'n_steps': 25},
    'past_len': 21, 'future_len': 60, 'seed': 42
}
attr_calculator = TrajAttrBase(model, config)

# 计算归因（配置参数自动传递）
attributions = attr_calculator.compute_attribution(batch, method='GuidedIG')

# 或者计算多种方法并保存（统一保存管理）
all_attributions = attr_calculator.compute_and_save_attribution(
    batch, methods=['GuidedIG', 'Saliency', 'Dirichlet']
)
```

### 2. 模型适配器直接使用

```python
from utils_attr.traj_attr.adapters import ModelAdapterFactory

# 手动创建适配器
adapter = ModelAdapterFactory.create_adapter(model, 'autobot')

# 查看支持的模型
print("支持的模型:", ModelAdapterFactory.get_supported_models())

# 测试适配器
attribution_inputs, static_inputs = adapter.get_attribution_inputs(batch), adapter.get_static_inputs(batch)
```

### 3. 不同模型的使用

```python
# AutoBot模型
autobot_config = {'model_name': 'autobot'}
autobot_attr = TrajAttrBase(autobot_model, autobot_config)

# Wayformer模型
wayformer_config = {'model_name': 'wayformer'}
wayformer_attr = TrajAttrBase(wayformer_model, wayformer_config)

# MTR模型
mtr_config = {'model_name': 'mtr'}
mtr_attr = TrajAttrBase(mtr_model, mtr_config)

# 相同的接口，不同的模型
for attr_calculator in [autobot_attr, wayformer_attr, mtr_attr]:
    attributions = attr_calculator.compute_attribution(batch, 'Saliency')
```

### 4. 运行时参数覆盖

```python
# 配置文件参数会自动传递，也可以运行时覆盖
dirichlet_attrs = attr_calculator.compute_attribution(
    batch,
    method='Dirichlet',
    alpha=0.05,      # 覆盖配置文件中的alpha=0.1
    n_paths=50,      # 覆盖配置文件中的n_paths=9
    n_steps=40       # 覆盖配置文件中的n_steps=25
)

# IntegratedGradients参数
ig_attrs = attr_calculator.compute_attribution(
    batch,
    method='IntegratedGradients',
    n_steps=100,
    method='gausslegendre'
)
```

## 归因方法调用链详解

### 核心调用流程

所有归因方法都遵循统一的调用流程，从入口点开始到最终输出结果：

```
入口脚本 -> 基础框架 -> 模型适配器 -> 归因方法 -> 距离计算 -> 结果保存
```

### 1. 完整调用链图

#### 主要入口点调用关系

```
compute_traj_attr.py::main()
├── TrajAttrExperiment.__init__()
├── TrajAttrExperiment.load_model_and_data()
├── TrajAttrExperiment.create_attributor()
│   └── TrajAttrBase(model, hydra_config)          # 直接传递DictConfig！
│       └── ModelAdapterFactory.create_adapter()   # adapters/model_adapters.py
└── TrajAttrExperiment.compute_batch_attributions()
    └── TrajAttrBase.compute_and_save_attribution()
        ├── TrajAttrBase.prepare_model_for_attribution()
        │   └── ModelAdapter.get_attribution_inputs()
        │   └── ModelAdapter.get_static_inputs()
        ├── TrajAttrBase.compute_attribution()       # 配置自动传递
        │   ├── [Dirichlet] config.get('dirichlet_config') → DirichletAttribution(**config)
        │   ├── [GuidedIG] config.get('guided_ig_config') → GuidedIGAttribution(**config)
        │   └── [Captum] config.get('captum_config') → CaptumAttribution(**config)
        └── TrajAttrBase.save_attribution_results()  # 统一保存管理
```

#### 归因计算核心循环

```
对每个batch:
  TrajAttrBase.compute_attribution() 
  ├── 准备输入: prepare_model_for_attribution()
  │   ├── adapter.get_attribution_inputs() -> Dict[str, Tensor]
  │   └── adapter.get_static_inputs() -> Dict[str, Any]
  ├── 方法选择和调用:
  │   ├── [method="Dirichlet"] -> DirichletAttribution.compute_attribution()
  │   ├── [method="GuidedIG"] -> GuidedIGAttribution.compute_attribution()  
  │   └── [method="IntegratedGradients"] -> CaptumAttribution.compute_attribution()
  └── 保存结果: save_attribution_results()
```

### 2. 各归因方法详细调用链

#### 🎯 Dirichlet方法调用链

```
DirichletAttribution.compute_attribution()                    # methods/dirichlet_attr.py
├── 对每个输入键(key):
│   ├── BaselineGenerator.generate_baseline()                 # utils_traj_attr/baseline_generator.py
│   ├── DirichletDistribution.sample()                       # 生成路径
│   ├── 对每条路径(path):
│   │   ├── DirichletAttribution._interpolate_path()
│   │   ├── DirichletAttribution._create_forward_wrapper()
│   │   │   └── ModelAdapter.forward_with_loss()             # adapters/model_adapters.py
│   │   │       ├── ModelAdapter.reconstruct_batch()
│   │   │       ├── model.forward()                          # 用户模型
│   │   │       ├── ModelAdapter.extract_prediction()
│   │   │       └── DistanceMetrics.get_distance_function()   # base/distance_metrics.py
│   │   └── torch.autograd.grad()                           # 梯度计算
│   └── DirichletAttribution._aggregate_attributions()       # 聚合路径归因
```

#### 🎯 Guided-IG方法调用链

```
GuidedIGAttribution.compute_attribution()                     # methods/guided_ig_attr.py
├── 对每个输入键(key):
│   ├── GuidedIGAttribution._create_forward_wrapper(target_input_key)
│   │   └── forward_func(inputs_tensor):
│   │       ├── 重构输入字典: 替换target_input_key对应的张量
│   │       └── ModelAdapter.forward_with_loss()             # adapters/model_adapters.py
│   ├── BaselineGenerator.generate_baseline()                # 生成基线
│   └── GuidedIGAttribution.attribute()                      # 主归因计算
│       ├── calculate_straight_line_path()                   # utils_traj_attr/tensor_utils.py
│       └── 对每个锚点(anchor):
│           └── GuidedIGAttribution.unbounded_guided_ig()
│               ├── l1_distance()                            # utils_traj_attr/tensor_utils.py
│               └── 对每个步骤(step):
│                   ├── GuidedIGAttribution._compute_gradients()
│                   │   └── torch.autograd.grad(forward_func(x), x)
│                   ├── 计算分位数阈值和集合S
│                   └── 更新归因值
```

#### 🎯 Captum方法调用链

```
CaptumAttribution.compute_attribution()                       # methods/captum_attr.py
├── CaptumAttribution._create_forward_wrapper()
│   └── forward_func(*input_tensors):
│       ├── 重构归因输入字典
│       ├── 处理batch_size不匹配问题
│       └── ModelAdapter.forward_with_loss()
├── CaptumAttribution.get_baseline()                         # 生成基线
│   └── BaselineGenerator.generate_baseline()
└── 调用具体Captum方法:
    ├── [IntegratedGradients] captum.attr.IntegratedGradients.attribute()
    ├── [Saliency] captum.attr.Saliency.attribute()  
    ├── [DeepLift] captum.attr.DeepLift.attribute()
    └── [其他方法] captum.attr.*.attribute()
        └── 内部调用用户定义的forward_func
```

### 3. 模型适配器调用详解

#### 模型适配器选择流程

```
ModelAdapterFactory.create_adapter(model, model_name)         # adapters/model_adapters.py
├── 自动检测模型类型:
│   ├── 检查model_name参数
│   ├── 检查模型类名(model.__class__.__name__)
│   └── 应用检测规则映射
├── 创建对应适配器:
│   ├── [AutoBot] AutoBotAdapter()
│   ├── [Wayformer] WayformerAdapter() 
│   ├── [MTR] MTRAdapter()
│   └── [默认] BaseModelAdapter()
└── 返回适配器实例
```

#### 适配器核心方法调用

```
对每个batch的处理:
ModelAdapter.get_attribution_inputs(batch)
├── 提取需要梯度的输入张量
├── 设置requires_grad=True
└── 返回Dict[str, Tensor]

ModelAdapter.get_static_inputs(batch)  
├── 提取掩码、索引等静态数据
└── 返回Dict[str, Any]

ModelAdapter.forward_with_loss(attribution_inputs, static_inputs, target_trajs)
├── ModelAdapter.reconstruct_batch()           # 重构模型输入格式
├── model.forward(reconstructed_batch)         # 模型前向传播
├── ModelAdapter.extract_prediction()          # 提取预测结果
├── DistanceMetrics.get_distance_function()    # 获取距离函数
└── distance_function(prediction, target_trajs) # 计算标量损失
```

### 4. 距离计算调用链

```
DistanceMetrics.get_distance_function(distance_type)          # base/distance_metrics.py
├── [distance_type="min_ade"] -> DistanceMetrics.min_ade_loss()
├── [distance_type="min_fde"] -> DistanceMetrics.min_fde_loss()
├── [distance_type="ade"] -> DistanceMetrics.ade_loss()
├── [distance_type="fde"] -> DistanceMetrics.fde_loss()
└── [其他] -> DistanceMetrics.l2_loss()

具体距离计算:
DistanceMetrics.min_ade_loss(pred_trajs, gt_trajs)
├── 计算所有模态的ADE: torch.norm(pred_trajs - gt_trajs, dim=-1)
├── 沿时间维度平均: distances.mean(dim=-1)
├── 选择最小距离模态: distances.min(dim=1)[0]  
└── 返回批次平均: distances.mean()
```

### 5. 结果保存调用链

```
TrajAttrBase.save_attribution_results(attributions, batch, method, metadata)
├── 对每个batch样本:
│   ├── 生成保存文件名
│   ├── 对每个输入的归因结果:
│   │   ├── utils_save.from_tensor_to_np()              # 转换为numpy
│   │   ├── np.save(path, attr_np)                      # 保存.npy文件
│   │   └── [可选] 生成可视化图像
│   └── 保存元数据信息
└── 输出保存路径信息
```

### 6. 配置文件加载调用链

```
compute_traj_attr.py使用Hydra配置系统:
hydra.main() -> main(cfg: DictConfig)
├── OmegaConf.merge(cfg, cfg.method)                    # 合并方法配置
├── OmegaConf.merge(cfg, cfg.attribution)              # 合并归因配置  
└── TrajAttrExperiment._create_attribution_config()
    ├── 提取guided_ig_config参数
    ├── 提取dirichlet_config参数
    ├── 提取captum_config参数
    └── 构建统一的attr_config字典

配置文件层次结构:
configs/traj_attr_base.yaml
├── defaults: [method/autobot, attribution/guided_ig, attribution/dirichlet, ...]
├── attribution.methods: ["GuidedIG", "Dirichlet", ...]
└── 其他基础配置

configs/attribution/guided_ig.yaml  
└── guided_ig_config: {steps, fraction, anchors, ...}
```

### 7. 实际运行示例和调用验证

#### 运行Guided-IG方法的完整调用流程

```bash
# 运行命令
cd exps_scripts/exp_trajattr/
python compute_traj_attr.py --config-name traj_attr_base
```

**实际调用日志示例：**
```
# 1. 入口和初始化阶段
compute_traj_attr.py::main() 
  └── [日志] "开始轨迹预测归因实验: traj_attr_exp"
  └── [日志] "模型: autobot, 数据集: nuscenes, 归因方法: ['GuidedIG']"

# 2. 模型和数据加载阶段  
TrajAttrExperiment.load_model_and_data()
  ├── build_model() -> AutoBot模型实例
  ├── [日志] "从检查点加载: TrajAttr_ckpt/autobot_train/best_model.ckpt"
  └── [日志] "数据集加载完成: 1000 个样本, 500 个批次"

# 3. 归因器创建阶段
TrajAttrExperiment.create_attributor()
  └── TrajAttrBase.__init__()
      ├── [日志] "创建归因计算配置..."
      ├── ModelAdapterFactory.create_adapter(model, 'autobot')
      │   └── [日志] "检测到AutoBot模型，使用AutoBotAdapter"
      └── [日志] "创建了 autobot 归因计算器（使用统一适配器）"

# 4. 批次处理阶段
TrajAttrExperiment.compute_batch_attributions()
  └── [日志] "开始计算归因，限制批次数: 3"
  └── 对每个batch (共3个):
      ├── [日志] "归因中 1/3"  
      └── TrajAttrBase.compute_and_save_attribution()
          ├── TrajAttrBase.prepare_model_for_attribution()
          │   ├── AutoBotAdapter.get_attribution_inputs()
          │   │   └── [返回] {'obj_trajs': Tensor[2,20,11], 'map_polylines': Tensor[2,50,9]}
          │   └── AutoBotAdapter.get_static_inputs()  
          │       └── [返回] {'center_gt_trajs': Tensor[2,60,2], 'obj_trajs_mask': ...}
          │
          ├── TrajAttrBase.compute_attribution(method='GuidedIG')
          │   └── [日志] "计算 GuidedIG 归因..."
          │   └── GuidedIGAttribution.compute_attribution()
          │       ├── [日志] "计算 obj_trajs 的 Guided-IG 归因..."
          │       ├── GuidedIGAttribution._create_forward_wrapper('obj_trajs')
          │       ├── BaselineGenerator.generate_baseline() -> 零基线
          │       └── GuidedIGAttribution.attribute()
          │           ├── calculate_straight_line_path(steps=10+1)
          │           └── 对每个锚点 (共10个):
          │               └── GuidedIGAttribution.unbounded_guided_ig()
          │                   └── 对每个步骤 (共2步):
          │                       ├── forward_func() -> loss值
          │                       ├── torch.autograd.grad() -> 梯度
          │                       └── 更新归因值
          │       ├── [日志] "计算 map_polylines 的 Guided-IG 归因..."  
          │       └── [重复上述过程]
          │
          └── TrajAttrBase.save_attribution_results()
              ├── [日志] "保存归因结果到: exps_res/res_trajattr/autobot_nuscenes/attributions/"
              ├── np.save("batch_0_obj_trajs_gig.npy", attr_np)
              └── np.save("batch_0_map_polylines_gig.npy", attr_np)

# 5. 完成阶段
  └── [日志] "归因计算完成，成功处理 3 个批次"
  └── [日志] "✓ 实验成功完成！结果保存在: exps_res/res_trajattr/autobot_nuscenes/"
```

#### 调用链验证方法

**1. 打开调试日志验证调用链：**
```python
# 在guided_ig_attr.py中添加调试信息
def _create_forward_wrapper(self, attribution_inputs, static_inputs, target_input_key):
    print(f"[DEBUG] 创建前向包装器，目标输入: {target_input_key}")
    
    def forward_func(inputs_tensor):
        print(f"[DEBUG] 前向传播，输入形状: {inputs_tensor.shape}")
        print(f"[DEBUG] 重构输入，替换 {target_input_key}")
        # ... 原有代码
        loss = self.attr_base.model_forward_wrapper(...)
        print(f"[DEBUG] 前向传播完成，损失: {loss.item()}")
        return loss
```

**2. 检查文件调用关系：**
```python
# 添加调用栈追踪  
import traceback

def compute_attribution(self, attribution_inputs, static_inputs, input_tensors):
    print("[TRACE] GuidedIGAttribution.compute_attribution() 被调用")
    print("[TRACE] 调用栈:")
    for line in traceback.format_stack()[-3:-1]:
        print(f"  {line.strip()}")
    # ... 原有代码
```

**3. 验证输入输出数据流：**
```python
# 在关键节点打印数据信息
def prepare_model_for_attribution(self, batch):
    attribution_inputs = self.model_adapter.get_attribution_inputs(batch)
    static_inputs = self.model_adapter.get_static_inputs(batch)
    
    print(f"[DATA] 归因输入键: {list(attribution_inputs.keys())}")  
    for key, tensor in attribution_inputs.items():
        print(f"[DATA] {key}: {tensor.shape}, requires_grad={tensor.requires_grad}")
    
    print(f"[DATA] 静态输入键: {list(static_inputs.keys())}")
    return attribution_inputs, static_inputs
```

### 支持的归因方法

所有归因方法遵循**单一职责原则**：只负责计算，不处理保存。

#### 1. Dirichlet方法 (自定义)
- **功能**: 基于Dirichlet分布的路径采样归因
- **配置**: `configs/attribution/dirichlet.yaml`
- **参数**: `alpha=0.1`, `n_paths=9`, `n_steps=25`
- **特点**: 适用于复杂非线性模型的归因计算

#### 2. Guided Integrated Gradients (GIG)
- **功能**: 无界引导积分梯度算法
- **配置**: `configs/attribution/guided_ig.yaml`  
- **参数**: `steps=50`, `fraction=0.1`, `anchors=10`
- **特点**: 针对每个输入分别计算，处理多输入模型

#### 3. Captum方法集成
- **功能**: 集成15+种经典归因方法
- **配置**: `configs/attribution/captum_methods.yaml`
- **支持方法**: 
  - `IntegratedGradients`: 积分梯度
  - `DeepLift`: DeepLift方法
  - `GradientShap`: 梯度SHAP
  - `Saliency`: 显著性图
  - `ShapleyValueSampling`: Shapley值采样
- **特点**: 成熟稳定，广泛验证

## 支持的距离度量

- `ade`: 平均位移误差
- `fde`: 最终位移误差  
- `min_ade`: 最小ADE (多模态)
- `min_fde`: 最小FDE (多模态)
- `l1`: L1距离
- `l2`: L2距离
- `smooth_l1`: 平滑L1距离

## 模型适配器系统

### 适配器架构

新的适配器系统通过统一接口自动处理不同模型的输入输出格式：

```python
class BaseModelAdapter:
    def get_attribution_inputs(self, batch):
        """提取需要计算归因的输入张量"""
        
    def get_static_inputs(self, batch):
        """提取不需要梯度的静态输入"""
        
    def reconstruct_batch(self, attribution_inputs, static_inputs):
        """重构模型可以接受的batch格式"""
        
    def extract_prediction(self, model_output):
        """从模型输出中提取标准化的预测结果"""
        
    def forward_with_loss(self, attribution_inputs, static_inputs, target_trajs):
        """执行前向传播并返回用于归因的标量损失"""
```

### 支持的模型适配器

#### 1. AutoBot适配器
- **输入适配**: `obj_trajs`, `map_polylines`等轨迹和地图数据
- **特征**: 自动提取轨迹特征（位置、速度等）和地图特征（点坐标、类型等）
- **输出**: 多模态轨迹预测和概率分布

#### 2. Wayformer适配器  
- **输入适配**: 兼容Perceiver架构的输入格式
- **特征**: ego-in, agents-in, roads等分层输入
- **输出**: 基于注意力机制的预测结果

#### 3. MTR适配器
- **输入适配**: 支持MTR特有的编码器-解码器结构
- **特征**: 多层次特征处理和场景编码
- **输出**: 精细化的多模态预测

#### 4. 通用适配器
- **自动检测**: 基于模型类名自动选择适配器
- **容错处理**: 无法识别的模型使用默认适配策略
- **扩展性**: 易于添加新模型的适配器

### 自动模型检测

```python
# 自动检测示例
adapter = ModelAdapterFactory.create_adapter(model)  # 自动检测
adapter = ModelAdapterFactory.create_adapter(model, 'autobot')  # 手动指定
```

检测规则：
- 根据模型类名（如`AutoBotEgo` -> `autobot`）
- 根据配置参数中的`model_name`
- 默认使用通用适配器作为备选

## 配置系统

### 新的模块化配置架构（推荐）

现在使用 Hydra 配置系统，支持模块化配置文件：

```yaml
# configs/traj_attr_base.yaml - 主配置
defaults:
  - method: autobot                    # 模型配置
  - attribution/dirichlet             # Dirichlet 方法配置
  - attribution/captum_methods        # Captum 方法配置  
  - attribution/guided_ig             # GuidedIG 方法配置

# 基础设置
model_name: "autobot"
dataset_name: "nuscenes" 
past_len: 21
future_len: 60
seed: 42

# 归因设置
attribution:
  enable: true
  methods: ["GuidedIG", "Dirichlet"]   # 支持的方法
  batch_limit: 3
  distance_type: "min_ade"

# 保存配置（统一管理）
save_config:
  base_dir: "exps_res/res_trajattr"
  save_formats: ["numpy", "json"]
```

```yaml
# configs/attribution/guided_ig.yaml - GIG 方法专用配置
guided_ig_config:
  steps: 50           # 积分步数
  fraction: 0.1       # 选择分位数的比例  
  anchors: 10         # 锚点数量
  baseline_type: "zero"
```

```yaml  
# configs/attribution/dirichlet.yaml - Dirichlet 方法专用配置
dirichlet_config:
  alpha: 0.1          # Dirichlet分布参数
  n_paths: 9          # 采样路径数量
  n_steps: 25         # 每条路径的步数
  baseline_type: "zero"
```

### 配置传递链路

```python
# 配置自动传递到对应方法
config.guided_ig_config   → GuidedIGAttribution(**guided_ig_config)
config.dirichlet_config   → DirichletAttribution(**dirichlet_config)
config.captum_config      → CaptumAttribution(**captum_config)
```

### 向后兼容的配置方式

```python
# 仍支持传统 dict 格式（自动检测）
config = {
    'model_name': 'autobot',
    'attr_methods': ['IntegratedGradients'], 
    'save_attr_results': True,
    'past_len': 21, 'future_len': 60
}
attr_calculator = TrajAttrBase(model, config)
```

## 输出结果

### 归因结果格式
```python
attributions = {
    'obj_trajs': torch.Tensor,      # 轨迹归因 [B, N, T, F]
    'map_polylines': torch.Tensor,  # 地图归因 [B, L, P, F]  
}
```

### 重要性分析
```python
analysis = {
    'temporal_importance': torch.Tensor,    # 时间重要性 [B, T]
    'agent_importance': torch.Tensor,       # 智能体重要性 [B, N]
    'feature_importance': torch.Tensor,     # 特征重要性 [B, F]
}
```

## 与evaluation_torch.py集成

在现有的evaluation_torch.py中添加归因计算:

```python
# 在main函数中添加
if cfg.get('enable_attribution', False):
    from utils_attr.traj_attr.evaluation.eval_integration import TrajAttrEvaluator
    
    evaluator = TrajAttrEvaluator(model, cfg, model_type=cfg.method.model_name)
    results = evaluator.evaluate_with_attribution(val_loader)
    
    print("评估和归因计算完成")
    return results['evaluation']['metrics']
else:
    # 原有的评估流程
    ...
```

## 扩展指南

### 添加新的模型适配器

1. **创建适配器类**：
```python
from utils_attr.traj_attr.adapters.model_adapters import BaseModelAdapter

class NewModelAdapter(BaseModelAdapter):
    def get_attribution_inputs(self, batch):
        """根据新模型的输入格式提取需要归因的张量"""
        inputs = {}
        input_dict = batch['input_dict']
        
        # 提取需要归因的输入，设置requires_grad=True
        if 'model_specific_input' in input_dict:
            inputs['model_specific_input'] = input_dict['model_specific_input'].detach().requires_grad_(True)
        
        return inputs
    
    def get_static_inputs(self, batch):
        """提取静态输入（不需要梯度的数据）"""
        inputs = {}
        input_dict = batch['input_dict']
        
        # 提取掩码、索引等静态数据
        for key in ['masks', 'indices', 'gt_data']:
            if key in input_dict:
                inputs[key] = input_dict[key]
        
        return inputs
    
    def reconstruct_batch(self, attribution_inputs, static_inputs):
        """重构模型期望的batch格式"""
        input_dict = {}
        input_dict.update(attribution_inputs)
        input_dict.update(static_inputs)
        
        return {'input_dict': input_dict}
    
    def extract_prediction(self, model_output):
        """提取预测结果"""
        prediction, loss = model_output
        return {
            'predicted_trajectory': prediction.get('pred_trajs'),
            'predicted_probability': prediction.get('pred_probs'),
            'loss': loss
        }
```

2. **注册新适配器**：
```python
from utils_attr.traj_attr.adapters import ModelAdapterFactory

# 注册新适配器
ModelAdapterFactory.register_adapter('new_model', NewModelAdapter)

# 使用新适配器
adapter = ModelAdapterFactory.create_adapter(model, 'new_model')
```

### 添加新的归因方法

1. **基于Captum的方法**：
   - 在`CaptumAttribution`类的`captum_methods`字典中添加新方法
   - 如果需要特殊参数处理，在`compute_attribution`中添加分支

2. **自定义方法**：
```python
class CustomAttribution:
    def __init__(self, attr_base, **kwargs):
        self.attr_base = attr_base
        # 初始化自定义参数
    
    def compute_attribution(self, attribution_inputs, static_inputs, input_tensors):
        """实现自定义归因算法"""
        # 实现归因计算逻辑
        attributions = {}
        for key in attribution_inputs.keys():
            # 计算每个输入的归因值
            attributions[key] = self.compute_single_attribution(
                attribution_inputs[key], static_inputs
            )
        return attributions

# 在TrajAttrBase中注册
def compute_attribution(self, batch, method='IntegratedGradients', **kwargs):
    if method == 'CustomMethod':
        from ..methods.custom_attr import CustomAttribution
        attr_calculator = CustomAttribution(self, **kwargs)
        return attr_calculator.compute_attribution(attribution_inputs, static_inputs, input_tensors)
```

### 添加新的距离度量

```python
# 在distance_metrics.py中添加
class DistanceMetrics:
    def custom_distance(self, pred_trajs, gt_trajs):
        """自定义距离计算"""
        # 实现自定义距离计算逻辑
        distance = torch.norm(pred_trajs - gt_trajs, p=1, dim=-1)  # 例如：L1距离
        return distance.mean()
    
    def get_distance_function(self, distance_type):
        """获取距离函数"""
        distance_functions = {
            # ... 现有的距离函数
            'custom': self.custom_distance,
        }
        return distance_functions.get(distance_type, self.min_ade_loss)
```

## 注意事项

1. **内存使用**: 归因计算需要额外的GPU内存，建议适当减小batch size
2. **计算时间**: 归因计算会显著增加评估时间，可通过`attribution_batch_limit`限制计算批次
3. **梯度计算**: 确保模型输入张量设置了`requires_grad=True`
4. **数据格式**: 确保输入数据格式与模型期望的格式匹配

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减小batch size或batch limit
2. **梯度计算错误**: 检查模型前向传播包装器
3. **维度不匹配**: 确认输入张量维度与模型期望一致

### 调试建议
- 使用小批次数据测试
- 检查模型输出格式
- 验证距离计算函数
- 启用详细日志输出

## 示例配置文件

使用配置:
```bash
python evaluation_torch.py +traj_attr=traj_attr_config
```