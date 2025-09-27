# 轨迹可视化脚本技术文档

本目录包含自动驾驶轨迹预测的可视化工具，为TrajAttrPub项目提供真实轨迹和预测轨迹的渲染功能。

## 概述

可视化系统使用MetaDrive仿真环境提供交互式自动驾驶场景渲染，允许研究人员同时可视化真实轨迹和模型预测结果。

## 核心组件

### 1. 基础可视化环境 (base_visualization.py)

扩展MetaDrive的ScenarioEnv，增强轨迹渲染功能的主要可视化环境。

主要特性：
- 场景加载：可配置的场景数据库路径
- 双重渲染：同时支持真实轨迹和预测轨迹可视化
- 交互控制：手动逐步执行场景，实时渲染
- 配置管理：基于YAML的配置，支持路径解析

使用方法：
```python
from base_visualization import BaseVisualizationEnv
env = BaseVisualizationEnv(config)
env.reset(seed=scenario_id)
while not done:
    env.step(action)
    env.render(mode="top_down")
```

### 2. 真实轨迹渲染器 (ground_truth_trajectory_renderer.py)

渲染场景数据中的实际车辆轨迹，对已完成、当前和未来轨迹段进行视觉区分。

功能特性：
- 轨迹提取：从场景元数据中自动识别目标车辆
- 多段可视化：对过去、当前和未来轨迹点使用不同颜色
- 自适应目标选择：支持多种轨迹识别方法
- 实时更新：逐帧轨迹进展可视化

视觉配置：
- 已完成段：绿色 (80, 200, 80)
- 当前位置：黄色圆圈，黑色边框
- 未来段：蓝色 (100, 150, 255)
- 线宽：3像素
- 当前标记半径：10像素

### 3. 预测轨迹渲染器 (prediction_trajectory_renderer.py)

渲染模型输出的预测轨迹，支持多模态预测和概率过滤。

功能特性：
- 多模态支持：为每个对象渲染多个预测模式
- 概率过滤：可配置的最小概率阈值
- JSON集成：从标准化JSON格式加载预测
- 虚线渲染：视觉上区分预测和真实轨迹
- 模式限制：可配置每个对象的最大模式数以提高清晰度

## 配置文件 (visualization_defaults.yaml)

```yaml
database_path: dataset_traj/scn_split_val
prediction_path: exps_scripts/unitraj_train_eval/predictions_autobot_world_coords.json
max_prediction_modes: 1
min_prediction_probability: 0.05
scenario_index: null
```

配置选项说明：
- database_path: 场景数据库路径（相对于项目根目录）
- prediction_path: 预测JSON文件路径
- max_prediction_modes: 限制每个对象显示的预测模式数
- min_prediction_probability: 过滤低置信度预测
- scenario_index: 渲染特定场景（null表示所有场景）

## 使用方式

### 基础可视化
```bash
cd vis_scripts
python base_visualization.py
```

### 自定义配置
1. 编辑 visualization_defaults.yaml
2. 设置所需的数据库和预测路径
3. 运行可视化脚本

### 与训练流水线集成
```python
# 模型训练和评估后
predictions = model.evaluate(test_data)
save_predictions("predictions_world_coords.json", predictions)

# 更新可视化配置
config = load_config("visualization_defaults.yaml")
config["prediction_path"] = "predictions_world_coords.json"

# 可视化结果
env = BaseVisualizationEnv(config)
visualize_scenarios(env)
```

## 技术架构

### 坐标系统
- 世界坐标：轨迹数据的全局坐标系
- 屏幕坐标：用于渲染的像素坐标
- 相机偏移：跟随目标车辆的动态相机

### 渲染流水线
1. 场景加载：加载场景数据并提取轨迹
2. 坐标变换：将世界坐标转换为屏幕像素
3. 可见性剔除：跳过屏幕外的轨迹点
4. 多层渲染：真实轨迹 → 预测 → 当前标记

## 依赖项

必需包：
- MetaDrive: 仿真环境和渲染后端
- PyGame: 底层图形和显示管理
- NumPy: 数值运算和数组处理
- PyYAML: 配置文件解析

系统要求：
- Python 3.7+
- OpenGL兼容的图形驱动
- 足够的RAM用于场景数据库

## 性能优化

优化特性：
- 可见性剔除：仅渲染可见的轨迹段
- 基于帧的更新：增量轨迹进展
- 可配置质量：可调整线宽和标记大小
- 内存管理：高效的轨迹数据结构

扩展建议：
- 对复杂场景限制 max_prediction_modes
- 使用 min_prediction_probability 过滤噪声
- 批量处理大型数据集的场景
- 考虑实时应用的GPU加速

## 故障排除

常见问题：
1. 缺少数据库：验证 database_path 存在且可读
2. 预测格式：确保JSON遵循预期模式
3. 显示问题：检查OpenGL和PyGame安装
4. 内存错误：减少场景批量大小或轨迹分辨率

调试配置：
- 在MetaDrive中启用详细日志
- 验证坐标变换
- 检查场景元数据提取
- 验证预测数据加载