
import os
import math
import re
import glob

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import utils_data.img_proc as img_proc

# Using Agg is much faster than nothing or TkAgg
import skimage
from skimage.transform import resize

import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d, axes3d

import matplotlib as mpl
import matplotlib.pyplot as plt

matplotlib.use('Agg')
# matplotlib.use('TkAgg')

# 设置全局字体  (推荐方法)
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定 SimHei 字体 (Windows)
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_exist_inv_dir(data_dir):
    file_names_temp = glob.glob(data_dir + "/*.png") + glob.glob(data_dir + "/*.jpg") + glob.glob(data_dir + "/*.jpeg")
    file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_names_temp]
    file_name_list = set()
    for file_name in file_names:
        first_dash_index = file_name.find("-")
        second_dash_index = file_name.find("-", first_dash_index + 1)
        img_name = file_name[:second_dash_index]
        file_name_list.add(img_name)
    file_name_list = list(file_name_list)
    return file_name_list


# 定义一个函数来提取字符串中的最后一个数字
def extract_last_number(input_string):
    match = re.search(r'\d+$', input_string)
    if match:
        return str(int(match.group()))
    else:
        return None


def get_decision(scores):
    """
    输入: torch tensor 得分, 尺寸: batch * 4
    输出: 字符串列表, 尺寸: batch * 4
    """
    decision_ori = scores >= 0.5
    decision_list = ['Forward', 'Break', 'Left', 'Right']
    
    result_list = []
    for row in decision_ori:
        decision_subset = [decision_list[i] if value else None for i, value in enumerate(row)]
        result_list.append(decision_subset)
    return result_list


def add_text(image_grid, decisions):
    width, height = image_grid.size
    bs = len(decisions)
    width_single = int(width / bs)
    if height > 3070:
        start_h = 2
        decision_start_height = 1
    else:
        start_h = 1.5
        decision_start_height = 0
    
    height_single = 258 * 2
    
    I1 = ImageDraw.Draw(image_grid)
    # Custom font style and font size
    myFont = ImageFont.truetype('FreeMono.ttf', 40)
    for i_bs in range(bs):
        text = ''
        for i in range(4):
            if decisions[i_bs][i] is not None:
                text = text + decisions[i_bs][i] + ','
        # Add Text to an image
        I1.text((10 + i_bs * width_single, 10 + decision_start_height * height_single), text,
                font=myFont, stroke_width=2, fill=(255, 0, 0))
    
    decision_list = ['NOT Forward', 'NOT Break', 'NOT Left', 'NOT Right']
    for i in range(4):
        I1.text((10, 10 + (i + start_h) * height_single), decision_list[i],
                font=myFont, stroke_width=2, fill=(255, 0, 0))
    return image_grid


def save_results(output_dir, bs, metadata, images, idx, flag_save_txt=True):
    images = {k: v.float().cpu() for k, v in images.items()}
    images = torch.cat([v for v in images.values()], 0)
    image_grid_tensor = make_grid(
        images, normalize=True, value_range=(-1, 1), nrow=bs
    )
    image_grid_no_text = F.to_pil_image(image_grid_tensor)
    image_grid_no_text = image_grid_no_text.save(f"{output_dir}/no_text_shot_{idx}.jpg")
    
    if flag_save_txt:
        image_grid = F.to_pil_image(image_grid_tensor)
        scores = metadata['init_scores']
        decisions = get_decision(scores)
        image_grid = add_text(image_grid, decisions)
        image_grid = image_grid.save(f"{output_dir}/shot_{idx}.jpg")
    
    torch.save(metadata, f'{output_dir}/metadata_{idx}.pt')


def plot_3d_weight_mesh(save_dir, save_name, weight,
                        figsize=(51.2, 25.6), flag_save=True):
    # 画出来blob score 的 3D 效果图, 像一座座山一样
    if torch.is_tensor(weight):
        weight = from_tensor_to_np(weight)
    if weight.shape[0] == 3 or weight.shape[0] == 1:
        weight = trans_img_channel(weight)
    
    # heatmap'size is same as input original image
    # plt.ioff()
    # plt.ion()
    fig = plt.figure(1, figsize=figsize, dpi=100, frameon=False)
    ax = fig.add_subplot(projection='3d')
    
    xx = np.arange(0.0, weight.shape[1], 1)
    yy = np.arange(0.0, weight.shape[0], 1)
    X, Y = np.meshgrid(xx, yy)
    
    # axis = plt.Axes(fig, [0., 0., 1., 1.])
    # axis.set_axis_off()
    # fig.add_axes(axis)
    
    # 将灰度值作为 Z 值
    Z = np.flip(weight, 0)
    
    # 绘制 3D 图形
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_axis_off()
    # ax.view_init(elev=4., azim=-45)
    
    set_axes_equal(ax)
    # fig.tight_layout()
    
    # 显示图形
    # plt.show()
    
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.png', transparent=True)
    
    plt.close(1)
    
    
def plot(save_dir, save_name, heatmap, ori_img,
         cmap='RdBu_r', cmap2='seismic', alpha=0.6,
         figsize=(5.12, 2.56), hm_range=None, flag_save=True):
    # alpha 越小, heatmap 覆盖能力越强 or 弱
    if torch.is_tensor(heatmap):
        heatmap = from_tensor_to_np(heatmap)
    if torch.is_tensor(ori_img):
        ori_img = from_tensor_to_np(ori_img)
    if heatmap.shape[0] == 3:
        heatmap = trans_img_channel(heatmap)
    if ori_img.shape[0] == 3:
        ori_img = trans_img_channel(ori_img)
    # if ori_img.dtype == np.uint8:
    #     ori_img = ori_img / 255
    
    # heatmap'size is same as input original image
    plt.ioff()
    # plt.ion()
    fig = plt.figure(1, figsize=figsize, dpi=100, frameon=False)
    
    # dx, dy = 0.05, 0.05
    # xx = np.arange(0.0, heatmap.shape[1] + dx, dx)
    # yy = np.arange(0.0, heatmap.shape[0] + dy, dy)
    # xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    # extent = xmin, xmax, ymin, ymax
    
    axis = plt.Axes(fig, [0., 0., 1., 1.])
    axis.set_axis_off()
    fig.add_axes(axis)
    
    cmap_xi = plt.get_cmap(cmap2).copy()
    cmap_xi.set_bad(alpha=0)
    overlay = ori_img
    if len(heatmap.shape) == 3:
        squeeze_hm(heatmap)
    
    # axis.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap,
    #             vmin=hm_range[0], vmax=hm_range[1])
    if hm_range is not None:
        axis.imshow(heatmap, interpolation='none', cmap=cmap,
                    vmin=hm_range[0], vmax=hm_range[1])
    else:
        axis.imshow(heatmap, interpolation='none', cmap=cmap)
    # axis.imshow(heatmap, interpolation='none', cmap=cmap)
    axis.imshow(overlay, interpolation='none', cmap=cmap_xi, alpha=alpha)
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.jpg')  # 'RdBu_r' 'hot'
    # else:
    #     plt.show()
    plt.close(1)
    # aaa = 1


def plot_grid_hm(save_dir, save_name, heatmap, ori_img, cmap='RdBu_r', cmap2='seismic', alpha=0.6,
                 figsize=(5.12, 2.56), hm_range=None, flag_save=True):
    # 计算每个热力图数据点对应图片上的像素范围
    heatmap_width_ratio = ori_img.shape[1] / heatmap.shape[1]
    heatmap_height_ratio = ori_img.shape[0] / heatmap.shape[0]
    
    # --------------------------------------------------------------------------
    # for 循环比较慢
    # # 创建一个新的数组来存储叠加后的图像
    # resize_hm = np.zeros((ori_img.shape[0], ori_img.shape[1]))
    # # 遍历热力图数据
    # for i in range(heatmap.shape[0]):
    #     for j in range(heatmap.shape[1]):
    #         # 计算当前热力图数据点对应图片上的像素范围
    #         start_row = int(i * heatmap_height_ratio)
    #         end_row = int((i + 1) * heatmap_height_ratio)
    #         start_col = int(j * heatmap_width_ratio)
    #         end_col = int((j + 1) * heatmap_width_ratio)
    #
    #         resize_hm[start_row:end_row, start_col:end_col] = heatmap[i, j]
    
    # --------------------------------------------------------------------------
    # 生成目标图像的坐标网格
    # 将热力图数据在纵横方向上重复扩展
    resize_hm = np.repeat(np.repeat(heatmap, int(heatmap_height_ratio), axis=0), int(heatmap_width_ratio), axis=1)
    # --------------------------------------------------------------------------
    
    plot(save_dir, save_name, resize_hm, ori_img,
         cmap=cmap, cmap2=cmap2, alpha=alpha,
         figsize=figsize, hm_range=hm_range, flag_save=flag_save)
    aaa = 1


def plot_lc_hm_basis(save_dir, save_name, heatmap, cmap='RdBu_r',
               hm_range=None, flag_save=True):
    # heatmap'size is same as input original image
    plt.ioff()
    fig = plt.figure(1, dpi=100, frameon=False)
    
    axis = plt.Axes(fig, [0., 0., 1., 1.])
    axis.set_axis_off()
    fig.add_axes(axis)
    
    if hm_range is not None:
        axis.imshow(heatmap, interpolation='none', cmap=cmap,
                    vmin=hm_range[0], vmax=hm_range[1])
    else:
        axis.imshow(heatmap, interpolation='none', cmap=cmap)
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.jpg')  # 'RdBu_r' 'hot'
    plt.close(1)


def plot_doplc_hm_no_text(save_dir, save_name, row_labels, col_labels,
                          heatmap, cmap='RdBu_r',
                          hm_range=None, flag_save=True,
                          font_family='sans-serif', font_size=13):
    """
    no text 主要意思是说每个 grid 里面都没有字, 主要为了生成英文版
    """
    
    # heatmap'size is same as input original image
    plt.ioff()
    if len(row_labels) == 1:
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 2.7), dpi=200)
        font_size = 12
    else:
        fig, ax = plt.subplots(figsize=(len(col_labels) * 1.2, len(row_labels) * 1.2), dpi=200)
        font_size = 25
    if hm_range is not None:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap,
                       vmin=hm_range[0], vmax=hm_range[1])
    else:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap)
    
    # 添加分割线
    for i in range(len(row_labels)):
        ax.axhline(i - 0.5, color='black', linewidth=1)  # 水平线
    for j in range(len(col_labels)):
        ax.axvline(j - 0.5, color='black', linewidth=1)  # 垂直线
    
    # -------------------------------------------------------------------
    # 下面两个代码块用于设定是否把横纵轴 tick 文字写上
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", family=font_family, size=font_size)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_yticklabels(), family=font_family, size=font_size+2)
    
    # -------------------------------------------------------------------
    # # 关闭 y 轴刻度和标签
    ax.set_yticks([])  #  这行代码清除 y 轴刻度
    # ax.set_xticks([])  #  这行代码清除 x 轴刻度
    
    # -------------------------------------------------------------------
    
    fig.tight_layout()
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.png', transparent=True)  # 'RdBu_r' 'hot'
    plt.close(fig)
    

def plot_doplc_hm_with_text(save_dir, save_name, row_labels, col_labels,
                            heatmap, cmap='RdBu_r',
                            hm_range=None, flag_save=True,
                            font_family='sans-serif', font_size=13):
    """
    with text 主要意思是说每个 grid 里面的数字也被写上了, 主要为了生成中文版参考
    """
    # heatmap'size is same as input original image
    plt.ioff()
    if len(row_labels) == 1:
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 2.7), dpi=200)
        font_size = 12
    else:
        fig, ax = plt.subplots(figsize=(len(col_labels) * 1.2, len(row_labels) * 1.2), dpi=200)

    if hm_range is not None:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap,
                       vmin=hm_range[0], vmax=hm_range[1])
    else:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap)
        
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", family=font_family, size=font_size)  # Apply font changes to x-axis labels
    plt.setp(ax.get_yticklabels(), family=font_family, size=font_size)  # Apply font changes to y-axis labels

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{heatmap[i, j]:.2f}",
                           ha="center", va="center", color="w",
                           family=font_family, size=font_size)  # Apply font changes to cell values
    
    fig.tight_layout()
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.jpg', transparent=True)  # 'RdBu_r' 'hot'
    plt.close(fig)


def plot_lstmlc_hm_chn(save_dir, save_name, row_labels, col_labels,
                       heatmap, cmap='RdBu_r',
                       hm_range=None, flag_save=True,
                       font_family='sans-serif', font_size=13):
    """
    主要为了生成中文版参考
    """
    # heatmap'size is same as input original image
    plt.ioff()
    fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 2.7), dpi=200)  # 调整长宽比例

    if hm_range is not None:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap,
                       vmin=hm_range[0], vmax=hm_range[1])
    else:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap)
    
    # ----------------------------------------------------------------
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", family=font_family, size=font_size)  # Apply font changes to x-axis labels
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_yticklabels(), family=font_family, size=font_size)  # Apply font changes to y-axis labels
    
    # ----------------------------------------------------------------
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{heatmap[i, j]:.2f}",
                           ha="center", va="center", color="w",
                           family=font_family, size=font_size)  # Apply font changes to cell values
        
    # ----------------------------------------------------------------
    
    # # 关闭 y 轴刻度和标签
    ax.set_yticks([])  #  这行代码清除 y 轴刻度
    # ax.set_xticks([])  #  这行代码清除 x 轴刻度
    # ----------------------------------------------------------------
    
    fig.tight_layout()
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.jpg', transparent=True)  # 'RdBu_r' 'hot'
    plt.close(fig)
    

def plot_lstmlc_hm_eng(save_dir, save_name, row_labels, col_labels,
                       heatmap, cmap='RdBu_r',
                       hm_range=None, flag_save=True,
                       font_family='sans-serif', font_size=18):
    """
    为了英文版, 放在论文里面的图
    有时字体不够大, 可以放大, 从 13 - 22 都有
    """
    # heatmap'size is same as input original image
    plt.ioff()
    
    flag_change_font = False
    if flag_change_font:
        font_size = 30
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 3.5), dpi=200)  # 调整长宽比例
    else:
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 2.7), dpi=200)  # 调整长宽比例
    
    if hm_range is not None:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap,
                       vmin=hm_range[0], vmax=hm_range[1])
    else:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap)
    
    # ----------------------------------------------------------------
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", family=font_family, size=font_size)  # Apply font changes to x-axis labels
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_yticklabels(), family=font_family, size=font_size)  # Apply font changes to y-axis labels
        
    # ----------------------------------------------------------------
    
    # # 关闭 y 轴刻度和标签
    ax.set_yticks([])  #  这行代码清除 y 轴刻度
    ax.set_xticks([])  #  这行代码清除 x 轴刻度
    # ----------------------------------------------------------------
    
    fig.tight_layout()
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.png', transparent=True)  # 'RdBu_r' 'hot'
    plt.close(fig)


def plot_dualtranslc_hm_chn(save_dir, save_name, row_labels, col_labels,
                            heatmap, cmap='RdBu_r',
                            hm_range=None, flag_save=True,
                            font_family='sans-serif', font_size=13):
    """
    主要为了生成中文版参考
    """
    # heatmap'size is same as input original image
    plt.ioff()
    fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 2.7), dpi=200)  # 调整长宽比例
    
    if hm_range is not None:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap,
                       vmin=hm_range[0], vmax=hm_range[1])
    else:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap)
    
    # ----------------------------------------------------------------
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", family=font_family, size=font_size)  # Apply font changes to x-axis labels
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_yticklabels(), family=font_family, size=font_size)  # Apply font changes to y-axis labels
    
    # ----------------------------------------------------------------
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{heatmap[i, j]:.2f}",
                           ha="center", va="center", color="w",
                           family=font_family, size=font_size)  # Apply font changes to cell values
    
    # ----------------------------------------------------------------
    
    # # 关闭 y 轴刻度和标签
    ax.set_yticks([])  # 这行代码清除 y 轴刻度
    # ax.set_xticks([])  #  这行代码清除 x 轴刻度
    # ----------------------------------------------------------------
    
    fig.tight_layout()
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.jpg', transparent=True)  # 'RdBu_r' 'hot'
    plt.close(fig)
    
    
def plot_dualtranslc_hm_eng(save_dir, save_name, row_labels, col_labels,
                            heatmap, cmap='RdBu_r',
                            hm_range=None, flag_save=True,
                            font_family='sans-serif', font_size=18):
    """
    为了英文版, 放在论文里面的图
    有时字体不够大, 可以放大, 从 13 - 22 都有
    """
    # heatmap'size is same as input original image
    plt.ioff()
    
    flag_change_font = True
    if flag_change_font:
        # font_size = 30
        font_size = 24
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 3.5), dpi=200)  # 调整长宽比例
    else:
        fig, ax = plt.subplots(figsize=(len(col_labels) * 0.8, len(row_labels) * 2.7), dpi=200)  # 调整长宽比例
    
    if hm_range is not None:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap,
                       vmin=hm_range[0], vmax=hm_range[1])
    else:
        im = ax.imshow(heatmap, interpolation='none', cmap=cmap)
    
    # ----------------------------------------------------------------
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", family=font_family, size=font_size)  # Apply font changes to x-axis labels
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_yticklabels(), family=font_family, size=font_size)  # Apply font changes to y-axis labels
    
    # ----------------------------------------------------------------
    
    # # # 关闭 y 轴刻度和标签
    ax.set_yticks([])  # 这行代码清除 y 轴刻度
    # ax.set_xticks([])  # 这行代码清除 x 轴刻度
    # ----------------------------------------------------------------
    
    fig.tight_layout()
    if flag_save:
        plt.savefig(save_dir + '/' + save_name + '.png', transparent=True)  # 'RdBu_r' 'hot'
    plt.close(fig)
    
    
def plot_arr(save_dir, save_name, arr, figsize):
    # 设置颜色映射
    cmap = plt.cm.RdBu_r  # 蓝-白-红
    
    fig = plt.figure(1, figsize=figsize, dpi=100, frameon=False)
    # 设置颜色范围
    vmin = np.min(arr)
    vmax = np.max(arr)
    
    axis = plt.Axes(fig, [0., 0., 1., 1.])
    axis.set_axis_off()
    fig.add_axes(axis)
    
    # 绘制热力图
    axis.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.colorbar()  # 添加颜色条
    
    plt.savefig(save_dir + '/' + save_name + '.jpg')
    plt.close(1)
    

def plot_cluster_attr(save_dir, save_name, X, labels,
                      n_clusters_, cluster_centers, figsize):
    plt.figure(1, figsize=figsize, dpi=100, frameon=False)
    plt.clf()
    
    colors = ["#dede00", "#377eb8", "#f781bf"]
    markers = ["x", "o", "^"]
    
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=5,
        )
    
    plt.savefig(save_dir + '/' + save_name + '.jpg')
    plt.close(1)
    
    
def cv2_plot(save_dir, save_name, heatmap, ori_img):
    if torch.is_tensor(heatmap):
        heatmap = from_tensor_to_np(heatmap)
    if torch.is_tensor(ori_img):
        ori_img = from_tensor_to_np(ori_img)
    if heatmap.shape[0] == 3:
        heatmap = trans_img_channel(heatmap)
    if ori_img.shape[0] == 3:
        ori_img = trans_img_channel(ori_img)
    
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 2)
    heatmap = to_255_img(heatmap)
    ori_img = to_255_img(ori_img)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.7, cv_img, 0.3, 0)
    
    cv2.imwrite(save_dir + '/' + save_name + '.jpg', super_imposed_img)
    aaa = 1


def save_tensor_img(save_dir, save_name, img_gen):
    # 把经过 make_grid 处理的 torch tensor 图像保存下来
    # img_gen = make_grid(img_gen, normalize=True, value_range=(-1, 1), nrow=1)
    image_grid_no_text = F.to_pil_image(img_gen)
    image_grid_no_text = image_grid_no_text.save(f"{save_dir}/{save_name}_test.jpg")


def save_np_rgb_img(save_dir, save_name, img):
    if img.shape[0] == 3:
        img = trans_img_channel(img)
    img = to_255_img(img)
    skimage.io.imsave(f"{save_dir}/{save_name}.png", img)


def save_np_01_img(save_dir, save_name, img):
    if img.shape[0] == 3:
        img = trans_img_channel(img)
    map_img = img * 255
    map_img = np.uint8(map_img)
    pil_image = Image.fromarray(map_img, mode='L')
    pil_image.save(f"{save_dir}/{save_name}.png")
    
    
def save_np_gray_img(save_dir, save_name, img):
    if img.shape[0] == 3:
        img = trans_img_channel(img)
    map_img = to_255_img(img)
    pil_image = Image.fromarray(map_img, mode='L')
    pil_image.save(f"{save_dir}/{save_name}.png")
    
    
def squeeze_hm(heatmap, flag="mean"):
    if flag == 'mean':
        heatmap = np.mean(heatmap, 2)
    else:
        heatmap = np.max(heatmap, 2)
    return heatmap


def to_01_arr(arr):
    normed_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return normed_arr


def to_11_arr(arr):
    # 负值部分缩放到-1到0, 而正值部分缩放到0到1
    # 将负值部分缩放到-1到0
    negative_scaled = np.interp(arr, (np.min(arr), 0), (-1, 0))
    # 将正值部分缩放到0到1
    positive_scaled = np.interp(arr, (0, np.max(arr)), (0, 1))
    # 合并缩放后的数组
    scaled_arr = np.where(arr < 0, negative_scaled, positive_scaled)
    return scaled_arr
    
    
def to_255_img(img):
    # 把 0-1 范围的图像转为 0-255 的 np 数组图像
    if np.max(img) < 3:
        normed_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        map_img = normed_img * 255
        map_img = np.uint8(map_img)
    else:
        map_img = np.uint8(img)
    return map_img


def from_tensor_to_np(tensor_ori):
    # 检查arr_torch的类型
    if type(tensor_ori) is torch.Tensor:
        tensor = tensor_ori.squeeze().cpu().detach().numpy()
    else:
        tensor = tensor_ori
    return tensor


def trans_img_channel(np_img):
    np_img = np.transpose(np_img, (1, 2, 0))
    return np_img


def from_tensor_img_to_np(tensor_ori):
    img_np = trans_img_channel(from_tensor_to_np(tensor_ori))
    img_np = to_255_img(img_np)
    return img_np


def trans_bgrimg_channel(np_img):
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img_rgb = np_img[..., ::-1]
    return np_img_rgb


def mean_to_255_img(img):
    map_img = img + [122.7717, 115.9465, 102.9801]
    map_img = np.clip(map_img, 0, 255)
    map_img = np.uint8(map_img)
    return map_img


def from_bgrtensor_img_to_np(tensor_ori):
    img_np = trans_bgrimg_channel(from_tensor_to_np(tensor_ori))
    img_np = mean_to_255_img(img_np)
    return img_np


def from_np_to_tensor(img_np, device):
    if img_np.shape[-1] == 3 or img_np.shape[-1] == 4:
        img_np = np.transpose(img_np, (2, 0, 1))
    img_np = img_np.astype(np.float32)
    if np.max(img_np) > 50:
        img_np = img_np / 255.
    img_t = torch.unsqueeze(torch.tensor(img_np, device=device), 0)
    return img_t


def from_np_to_int_tensor(img_np):
    if img_np.shape[-1] == 3 or img_np.shape[-1] == 4:
        img_np = np.transpose(img_np, (2, 0, 1))
    img_t = torch.unsqueeze(torch.tensor(img_np, dtype=torch.uint8), 0)
    return img_t


def read_img_and_resize(img_path, resolution):
    color_mode1 = cv2.COLOR_BGR2RGB
    img = img_proc.read_proc_img(img_path, color_mode1)
    img = to_255_img(resize(img, resolution))
    return img


def blend_img(img, blob_img, alpha):
    # 需要两个图像都是处理为 0-255 的 np 数组
    # alpha 越小, blob 图越明显
    if img.shape[-1] == 4:
        img = img[..., :3]
    blended_image = alpha * img + (1 - alpha) * blob_img
    return blended_image


def set_axes_equal(ax):
    """
    在画 3D 图时,使每个轴都有相同而缩放比例
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    
    x_limits = ax.get_xlim3d()
    # x_limits = (0, 350)
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def sort_np_arr(arr):
    sorted_indices = np.argsort(arr)
    sorted_indices = sorted_indices[::-1]
    res = arr[sorted_indices]
    return res, sorted_indices
