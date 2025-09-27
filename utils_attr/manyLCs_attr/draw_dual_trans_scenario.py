import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def draw_vehicle_scenario(data, metadata, save_dir=None, file_name=None, use_chinese=True):
    """
    绘制车辆场景图，显示目标车辆及其周围的车辆

    参数:
        data: 包含车辆位置和速度信息的字典或数组
            格式: {
                'F-Dist': 正前车距离, 'F-Vel': 正前车相对速度,
                'LF-Dist': 左前车距离, 'LF-Vel': 左前车相对速度,
                'LB-Dist': 左后车距离, 'LB-Vel': 左后车相对速度,
                'RF-Dist': 右前车距离, 'RF-Vel': 右前车相对速度,
                'RB-Dist': 右后车距离, 'RB-Vel': 右后车相对速度
            }
            或者是包含10个元素的数组 [F-Dist, F-Vel, LF-Dist, LF-Vel, LB-Dist, LB-Vel, RF-Dist, RF-Vel, RB-Dist, RB-Vel]
        metadata: 元数据，包含车辆ID和帧号等信息
        save_dir: 保存目录，如果为None则不保存
        file_name: 文件名，如果为None则使用默认名称
        use_chinese: 是否使用中文标签

    返回:
        fig, ax: matplotlib的图形对象
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 转换数据格式（如果是数组）
    if isinstance(data, (list, tuple, np.ndarray)):
        data_dict = {
            'F-Dist': data[0], 'F-Vel': data[1],
            'LF-Dist': data[2], 'LF-Vel': data[3],
            'LB-Dist': data[4], 'LB-Vel': data[5],
            'RF-Dist': data[6], 'RF-Vel': data[7],
            'RB-Dist': data[8], 'RB-Vel': data[9]
        }
    else:
        data_dict = data
    
    # 提取车辆ID和帧号
    vehicle_id = metadata['other_info'][0] if 'other_info' in metadata else "unknown"
    frame_start = metadata['other_info'][1] if 'other_info' in metadata and len(metadata['other_info']) > 1 else 0
    frame_end = metadata['other_info'][2] if 'other_info' in metadata and len(metadata['other_info']) > 2 else 0
    
    # 默认车辆尺寸
    car_length = 5  # 车长
    car_width = 2  # 车宽
    
    # 计算需要的X轴范围，确保能显示所有车辆
    # 对于前车距离: 由于F-Dist = 自车 - 前车，因此前车实际在 -F-Dist 位置
    # 对于后车距离: 由于LB-Dist/RB-Dist = 后车 - 自车，因此后车实际在 LB-Dist/RB-Dist 位置
    max_forward_dist = max(
        [abs(data_dict['F-Dist']) if abs(data_dict['F-Dist']) < 999.0 else 0,
         abs(data_dict['LF-Dist']) if abs(data_dict['LF-Dist']) < 999.0 else 0,
         abs(data_dict['RF-Dist']) if abs(data_dict['RF-Dist']) < 999.0 else 0]) * 1.2  # 增加20%的边距
    
    max_backward_dist = max(
        [abs(data_dict['LB-Dist']) if abs(data_dict['LB-Dist']) < 999.0 else 0,
         abs(data_dict['RB-Dist']) if abs(data_dict['RB-Dist']) < 999.0 else 0]) * 1.2  # 增加20%的边距
    
    # 确保最小显示范围
    max_forward_dist = max(max_forward_dist, 100)  # 前方至少100米
    max_backward_dist = max(max_backward_dist, 100)  # 后方至少100米
    
    # 设置图形大小（使更宽长）
    fig, ax = plt.subplots(figsize=(20, 6))  # 更宽的图形尺寸
    
    # 设置坐标范围，以目标车辆为中心
    ax.set_xlim(-max_backward_dist, max_forward_dist)
    ax.set_ylim(-8, 8)  # 横向范围稍微缩小，使道路更细长
    
    # 绘制三条车道线，贯穿整个图形
    for y in [-4, 0, 4]:
        if y == 0:  # 中间车道用实线
            ax.axhline(y=y, color='black', linestyle='-', linewidth=2)
        else:  # 其他车道用虚线
            ax.axhline(y=y, color='black', linestyle='--', linewidth=1)
    
    # 添加坐标轴和方向标识
    ax.set_xlabel('X (行驶方向)', fontsize=12)
    ax.set_ylabel('Y (横向)', fontsize=12)
    
    ax.arrow(-max_backward_dist * 0.9, -5, 10, 0, head_width=0.5, head_length=2, fc='black', ec='black')
    ax.text(-max_backward_dist * 0.9 + 12, -5, 'X', fontsize=12)
    ax.arrow(-max_backward_dist * 0.9, -5, 0, 2, head_width=0.5, head_length=1, fc='black', ec='black')
    ax.text(-max_backward_dist * 0.9 - 1, -2, 'Y', fontsize=12)
    
    # 添加距离刻度线
    for x in range(-int(max_backward_dist), int(max_forward_dist) + 1, 20):
        if x != 0:  # 不在自车位置添加刻度
            ax.axvline(x=x, color='gray', linestyle=':', linewidth=0.5)
            ax.text(x, -7.5, f"{x}m", fontsize=8, ha='center', va='top', color='gray')
    
    # 绘制自车（蓝色）
    target_car = patches.Rectangle((-car_length / 2, -car_width / 2), car_length, car_width,
                                   color='lightblue', alpha=0.8, ec='black')
    ax.add_patch(target_car)
    ax.text(0, 0, '自车', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 绘制箭头表示自车的速度
    ax.arrow(car_length / 2, 0, 5, 0, head_width=0.5, head_length=1, fc='black', ec='black')
    
    # 车辆状态标志（实线表示存在的车辆，虚线表示虚拟车辆）
    DEFAULT_DISTANCE = 999.0
    
    # 绘制正前方车辆
    if abs(data_dict['F-Dist']) < DEFAULT_DISTANCE:
        # 注意：F-Dist = 自车 - 前车，所以前车位置是 -F-Dist
        x_front = data_dict['F-Dist']   # 取负值，使前车位于前方
        front_car = patches.Rectangle((x_front - car_length / 2, -car_width / 2), car_length, car_width,
                                      color='cyan', alpha=0.8, ec='black')
        ax.add_patch(front_car)
        ax.text(x_front, 0, '前车', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 显示相对速度
        speed_text = f"{data_dict['F-Vel']:.1f} m/s"
        ax.text(x_front, -car_width / 2 - 1, speed_text, ha='center', va='top', fontsize=10)
        
        # 显示距离
        dist_line = ax.annotate('', xy=(x_front - car_length / 2, -2), xytext=(car_length / 2, -2),
                                arrowprops=dict(arrowstyle='<->', color='black'))
        ax.text((x_front - car_length / 2 + car_length / 2) / 2, -2.5, f'{abs(data_dict["F-Dist"]):.1f}m',
                ha='center', fontsize=10)
    else:
        # 虚拟车辆（用虚线表示）
        x_front = max_forward_dist * 0.8  # 放在远处
        front_car = patches.Rectangle((x_front - car_length / 2, -car_width / 2), car_length, car_width,
                                      color='cyan', alpha=0.4, ec='black', linestyle='--')
        ax.add_patch(front_car)
        ax.text(x_front, 0, '前车(虚拟)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制左前方车辆
    if abs(data_dict['LF-Dist']) < DEFAULT_DISTANCE:
        # 注意：LF-Dist = 自车 - 左前车，所以左前车位置是 -LF-Dist
        x_left_front = data_dict['LF-Dist']
        left_front_car = patches.Rectangle((x_left_front - car_length / 2, 4 - car_width / 2), car_length, car_width,
                                           color='lightgray', alpha=0.8, ec='black')
        ax.add_patch(left_front_car)
        ax.text(x_left_front, 4, '左前车', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 显示相对速度
        speed_text = f"{data_dict['LF-Vel']:.1f} m/s"
        ax.text(x_left_front, 4 - car_width / 2 - 1, speed_text, ha='center', va='top', fontsize=10)
        
        # 显示距离
        # 绘制一条从自车左前角到左前车右后角的线
        dist_line = ax.annotate('', xy=(x_left_front - car_length / 2, 4 - car_width / 2),
                                xytext=(car_length / 2, car_width / 2),
                                arrowprops=dict(arrowstyle='<->', color='black', linestyle=':'))
        
        # 计算线的中点位置
        midx = (x_left_front - car_length / 2 + car_length / 2) / 2
        midy = (4 - car_width / 2 + car_width / 2) / 2
        ax.text(midx, midy, f'{abs(data_dict["LF-Dist"]):.1f}m',
                ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    else:
        # 虚拟车辆
        x_left_front = max_forward_dist * 0.8
        left_front_car = patches.Rectangle((x_left_front - car_length / 2, 4 - car_width / 2), car_length, car_width,
                                           color='lightgray', alpha=0.4, ec='black', linestyle='--')
        ax.add_patch(left_front_car)
        ax.text(x_left_front, 4, '左前车(虚拟)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制左后方车辆
    if abs(data_dict['LB-Dist']) < DEFAULT_DISTANCE:
        # 注意：LB-Dist = 左后车 - 自车，所以左后车位置是 LB-Dist (已经是正确方向)
        x_left_behind = data_dict['LB-Dist']
        left_behind_car = patches.Rectangle((x_left_behind - car_length / 2, 4 - car_width / 2), car_length, car_width,
                                            color='orange', alpha=0.8, ec='black')
        ax.add_patch(left_behind_car)
        ax.text(x_left_behind, 4, '左后车', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 显示相对速度
        speed_text = f"{data_dict['LB-Vel']:.1f} m/s"
        ax.text(x_left_behind, 4 - car_width / 2 - 1, speed_text, ha='center', va='top', fontsize=10)
        
        # 显示距离
        # 绘制一条从自车左后角到左后车右前角的线
        dist_line = ax.annotate('', xy=(x_left_behind + car_length / 2, 4 - car_width / 2),
                                xytext=(-car_length / 2, car_width / 2),
                                arrowprops=dict(arrowstyle='<->', color='black', linestyle=':'))
        
        # 计算线的中点位置
        midx = (x_left_behind + car_length / 2 - car_length / 2) / 2
        midy = (4 - car_width / 2 + car_width / 2) / 2
        ax.text(midx, midy, f'{abs(data_dict["LB-Dist"]):.1f}m',
                ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    else:
        # 虚拟车辆
        x_left_behind = -max_backward_dist * 0.7
        left_behind_car = patches.Rectangle((x_left_behind - car_length / 2, 4 - car_width / 2), car_length, car_width,
                                            color='orange', alpha=0.4, ec='black', linestyle='--')
        ax.add_patch(left_behind_car)
        ax.text(x_left_behind, 4, '左后车(虚拟)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制右前方车辆
    if abs(data_dict['RF-Dist']) < DEFAULT_DISTANCE:
        # 注意：RF-Dist = 自车 - 右前车，所以右前车位置是 -RF-Dist
        x_right_front = data_dict['RF-Dist']
        right_front_car = patches.Rectangle((x_right_front - car_length / 2, -4 - car_width / 2), car_length, car_width,
                                            color='lightgray', alpha=0.8, ec='black')
        ax.add_patch(right_front_car)
        ax.text(x_right_front, -4, '右前车', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 显示相对速度
        speed_text = f"{data_dict['RF-Vel']:.1f} m/s"
        ax.text(x_right_front, -4 - car_width / 2 - 1, speed_text, ha='center', va='top', fontsize=10)
        
        # 显示距离
        # 绘制一条从自车右前角到右前车左后角的线
        dist_line = ax.annotate('', xy=(x_right_front - car_length / 2, -4 + car_width / 2),
                                xytext=(car_length / 2, -car_width / 2),
                                arrowprops=dict(arrowstyle='<->', color='black', linestyle=':'))
        
        # 计算线的中点位置
        midx = (x_right_front - car_length / 2 + car_length / 2) / 2
        midy = (-4 + car_width / 2 - car_width / 2) / 2
        ax.text(midx, midy, f'{abs(data_dict["RF-Dist"]):.1f}m',
                ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    else:
        # 虚拟车辆
        x_right_front = max_forward_dist * 0.8
        right_front_car = patches.Rectangle((x_right_front - car_length / 2, -4 - car_width / 2), car_length, car_width,
                                            color='lightgray', alpha=0.4, ec='black', linestyle='--')
        ax.add_patch(right_front_car)
        ax.text(x_right_front, -4, '右前车(虚拟)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制右后方车辆
    if abs(data_dict['RB-Dist']) < DEFAULT_DISTANCE:
        # 注意：RB-Dist = 右后车 - 自车，所以右后车位置是 RB-Dist (已经是正确方向)
        x_right_behind = data_dict['RB-Dist']
        right_behind_car = patches.Rectangle((x_right_behind - car_length / 2, -4 - car_width / 2), car_length,
                                             car_width,
                                             color='orange', alpha=0.8, ec='black')
        ax.add_patch(right_behind_car)
        ax.text(x_right_behind, -4, '右后车', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 显示相对速度
        speed_text = f"{data_dict['RB-Vel']:.1f} m/s"
        ax.text(x_right_behind, -4 - car_width / 2 - 1, speed_text, ha='center', va='top', fontsize=10)
        
        # 显示距离
        # 绘制一条从自车右后角到右后车左前角的线
        dist_line = ax.annotate('', xy=(x_right_behind + car_length / 2, -4 + car_width / 2),
                                xytext=(-car_length / 2, -car_width / 2),
                                arrowprops=dict(arrowstyle='<->', color='black', linestyle=':'))
        
        # 计算线的中点位置
        midx = (x_right_behind + car_length / 2 - car_length / 2) / 2
        midy = (-4 + car_width / 2 - car_width / 2) / 2
        ax.text(midx, midy, f'{abs(data_dict["RB-Dist"]):.1f}m',
                ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    else:
        # 虚拟车辆
        x_right_behind = -max_backward_dist * 0.7
        right_behind_car = patches.Rectangle((x_right_behind - car_length / 2, -4 - car_width / 2), car_length,
                                             car_width,
                                             color='orange', alpha=0.4, ec='black', linestyle='--')
        ax.add_patch(right_behind_car)
        ax.text(x_right_behind, -4, '右后车(虚拟)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 添加标题
    lane_change_type = "左侧换道" if metadata.get('label') == 1 else "右侧换道" if metadata.get(
        'label') == 2 else "保持车道"
    title = f"车辆ID: {vehicle_id}, 帧: {frame_start}-{frame_end}, 行为: {lane_change_type}"
    ax.set_title(title, fontsize=14)
    
    # 添加图例说明
    legend_text = "相对速度标注在各车下方，相对距离标注在连线上"
    plt.figtext(0.5, 0.01, legend_text, fontsize=12, ha='center')
    
    # 添加水平和垂直网格线以便于参考
    ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)  # 为底部的图例留出空间
    
    # 保存图像
    if save_dir is not None:
        if file_name is None:
            if 'attr_hm_name' in metadata:
                file_name = f"scenario_{metadata['attr_hm_name']}.png"
            else:
                file_name = f"scenario_vehicle_{vehicle_id}_frame_{frame_start}_{frame_end}.png"
        
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"场景图已保存至: {save_path}")
    
    return fig, ax