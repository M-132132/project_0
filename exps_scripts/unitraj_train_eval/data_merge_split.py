import os
import subprocess
import shutil
import re
import math
from pathlib import Path
from utils.path_manager import path_manager

# ---------------- 配置部分 ----------------

# 源数据集的根目录（绝对路径，因为数据集不在项目里）
base_dir = r"D:\datasets\nuscenes_scn\nuscenes_Traj"

# 项目目录下的 data 文件夹（相对路径）
data_dir = path_manager.root_dir / "dataset_traj"
data_dir.mkdir(exist_ok=True)

# 输出目录（相对路径，存放在项目 data 下）
out_dir = str(data_dir / "scn_merged")
train_dir = str(data_dir / "scn_split_train")
val_dir   = str(data_dir / "scn_split_val")

# 想屏蔽的数据集名字（文件夹名的一部分即可）
exclude = ["Argoverse", "waymo"]   # 例如 ["v1.0-mini", "waymo_scn"]

# 训练集比例
train_ratio = 0.8


# ---------------- 功能函数 ----------------

def get_scenario_num(dir_path: str) -> int:
    """调用官方命令 `scenarionet.num` 获取场景数量"""
    cmd = ["python", "-m", "scenarionet.num", "-d", dir_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="ignore")
    print(res.stdout)   # 打印官方输出
    matches = re.findall(r"Number of scenarios:\s*(\d+)", res.stdout)
    if matches:
        return int(matches[-1])
    else:
        raise RuntimeError("未能从 scenarionet.num 输出解析到场景数")


# ---------------- 主流程 ----------------

# 搜索所有已转换好的数据库路径
db_paths = []
for root, dirs, files in os.walk(base_dir):
    if "dataset_summary.pkl" in files:
        if any(ex in root for ex in exclude):
            continue
        db_paths.append(root)

print("找到的数据库：")
for p in db_paths:
    print(" ", p)

# 如果输出目录已存在，删除
if os.path.exists(out_dir):
    print(f"⚠️ 检测到 {out_dir} 已存在，正在删除...")
    shutil.rmtree(out_dir)

# 合并数据库
cmd_merge = ["python", "-m", "scenarionet.merge", "--from"] + db_paths + ["-d", out_dir]
print("\n执行命令：", " ".join(cmd_merge))
subprocess.run(cmd_merge, text=True, encoding="utf-8", errors="ignore")

# 用官方命令获取场景数
merged_total = get_scenario_num(out_dir)

# 按比例计算训练/验证集数量
train_num = int(math.floor(merged_total * train_ratio + 0.5))
val_num = merged_total - train_num
print(f"[切分] 合并后共 {merged_total} 个场景 → 训练 {train_num}，验证 {val_num}")

# 清理已有切分目录
for d in [train_dir, val_dir]:
    if os.path.exists(d):
        print(f"⚠️ 检测到 {d} 已存在，正在删除...")
        shutil.rmtree(d)

# 生成训练集
cmd_split_train = [
    "python", "-m", "scenarionet.split",
    "--from", out_dir,
    "--to", train_dir,
    "--num_scenarios", str(train_num)
]
print("执行命令：", " ".join(cmd_split_train))
subprocess.run(cmd_split_train, text=True, encoding="utf-8", errors="ignore")

# 生成验证集
cmd_split_val = [
    "python", "-m", "scenarionet.split",
    "--from", out_dir,
    "--to", val_dir,
    "--num_scenarios", str(val_num),
    "--start_index", str(train_num)
]
print("执行命令：", " ".join(cmd_split_val))
subprocess.run(cmd_split_val, text=True, encoding="utf-8", errors="ignore")

print(" 完成 ，数据集在data文件夹 ")
