
import numpy as np
import cv2

import utils_data.img_proc as img_proc
import utils_data.utils_save as utils_save


def save_attr_img_hm(save_dir, attr_index_str, attrs, img, attr_method=None, attrs_ori=None, figsize=None):
    ori_save_name = 'img_re'
    
    if attr_method is None:
        attr_hm_save_name = 'attrHM-{}'.format(attr_index_str)
        attr_ori_save_name = 'attr-{}'.format(attr_index_str)
    else:
        attr_hm_save_name = 'attrHM-{}-{}'.format(attr_method, attr_index_str)
        attr_ori_save_name = 'attr-{}-{}'.format(attr_method, attr_index_str)
        
    attr_range = (np.min(attrs), np.max(attrs))
    
    # 保存原始图和全部归因的热力图
    utils_save.save_np_rgb_img(save_dir, ori_save_name, img)
    utils_save.plot(save_dir, attr_hm_save_name,
                    attrs, img, alpha=0.15, figsize=figsize,
                    hm_range=attr_range, flag_save=True)
    if attrs_ori is not None:
        # 保存的是原始归因图, 有正有负的
        utils_save.plot_arr(save_dir, attr_ori_save_name, attrs_ori, figsize)
    aaa = 1
    

def save_nattr_img_hm(save_dir, heatmap_name, resized_attr_hm, img,
                      attrs_filter=None, figsize=None):
    attr_range = (np.min(resized_attr_hm), np.max(resized_attr_hm))
    if attrs_filter is not None:
        # 保存的是原始归因图, 有正有负的
        # utils_save.plot_arr(save_dir, heatmap_name+'-orihm', attrs_ori, figsize)
        # attr_range2 = (np.min(attrs_ori), np.max(attrs_ori))
        utils_save.plot_grid_hm(save_dir, heatmap_name+'-ori',
                                attrs_filter, img, alpha=0.15, figsize=figsize,
                                hm_range=attr_range, flag_save=True)
    else:
        # 保存原始图和全部归因的热力图
        utils_save.plot(save_dir, heatmap_name,
                        resized_attr_hm, img, alpha=0.15, figsize=figsize,
                        hm_range=attr_range, flag_save=True)
    aaa = 1
    

def save_doplc_attr_hm(save_dir, hm_name, row_labels, col_labels, attrs, attrs_filter, flag_chn_text):
    """
    保存的通常是 8*7 的 DOP 类型的矩阵
    """
    attr_range = (np.min(attrs_filter), np.max(attrs_filter))
    if flag_chn_text:
        utils_save.plot_doplc_hm_with_text(save_dir, hm_name, row_labels=row_labels, col_labels=col_labels, heatmap=attrs, hm_range=attr_range)
    else:
        utils_save.plot_doplc_hm_no_text(save_dir, hm_name, row_labels=row_labels, col_labels=col_labels, heatmap=attrs, hm_range=attr_range)
    aaa = 1


def save_lstmlc_attr_hm(save_dir, hm_name, row_labels, col_labels, attrs, attrs_filter, flag_chn_text):
    """
    保存的是 50*16 的矩阵, 需要给 50 压扁
    """
    attr_range = (np.min(attrs_filter), np.max(attrs_filter))
    if flag_chn_text:
        utils_save.plot_lstmlc_hm_chn(save_dir, hm_name, row_labels=row_labels, col_labels=col_labels, heatmap=attrs, hm_range=attr_range)
    else:
        utils_save.plot_lstmlc_hm_eng(save_dir, hm_name, row_labels=row_labels, col_labels=col_labels, heatmap=attrs, hm_range=attr_range)
    aaa = 1


def save_duantranslc_attr_hm(save_dir, hm_name, row_labels, col_labels, attrs, attrs_filter, flag_chn_text):
    """
    保存的是 180*10 的矩阵, 需要给 180 压扁
    """
    attr_range = (np.min(attrs_filter), np.max(attrs_filter))
    if flag_chn_text:
        utils_save.plot_dualtranslc_hm_chn(save_dir, hm_name, row_labels=row_labels, col_labels=col_labels, heatmap=attrs, hm_range=attr_range)
    else:
        utils_save.plot_dualtranslc_hm_eng(save_dir, hm_name, row_labels=row_labels, col_labels=col_labels, heatmap=attrs, hm_range=attr_range)
    aaa = 1
    

def save_grad_img_hm(save_dir, heatmap_name, grad_quan, img,
                     grad=None, figsize=None):
    attr_range = (np.min(grad_quan), np.max(grad_quan))
    # 保存的是原始归因图, 有正有负的
    # utils_save.plot_arr(save_dir, heatmap_name+'-orihm', attrs_ori, figsize)
    # attr_range2 = (np.min(attrs_ori), np.max(attrs_ori))
    utils_save.plot_grid_hm(save_dir, heatmap_name, grad, img, alpha=0.15, figsize=figsize,
                            hm_range=attr_range, flag_save=True)
    aaa = 1
    
    
def save_nattr_img_hm_res(save_dir, heatmap_name, attrs, img,
                          attrs_ori=None, figsize=None):
    attr_range = (np.min(attrs), np.max(attrs))
    
    # # 保存原始图和全部归因的热力图
    # utils_save.save_np_rgb_img(save_dir, save_img_name, img)
    
    # utils_save.plot(save_dir, heatmap_name,
    #                 attrs, img, alpha=0.15, figsize=figsize,
    #                 hm_range=attr_range, flag_save=True)
    utils_save.plot_grid_hm(save_dir, heatmap_name, attrs, img, alpha=0.15, figsize=figsize,
                            hm_range=attr_range, flag_save=True)
    
    # if attrs_ori is not None:
    #     # 保存的是原始归因图, 有正有负的
    #     # utils_save.plot_arr(save_dir, heatmap_name+'-orihm', attrs_ori, figsize)
    #     # attr_range2 = (np.min(attrs_ori), np.max(attrs_ori))
    #     utils_save.plot_grid_hm(save_dir, heatmap_name+'-ori', attrs_ori, img, alpha=0.15, figsize=figsize,
    #                             hm_range=attr_range, flag_save=True)
    aaa = 1
    
    
def save_masked_attr_img_hm(save_dir, save_name, attrs, img, figsize, attr_range=None):
    if attr_range is None:
        attr_range = (np.min(attrs), np.max(attrs))
    # 保存原始图和全部归因的热力图
    utils_save.plot(save_dir, save_name,
                    attrs, img, alpha=0.15, figsize=figsize,
                    hm_range=attr_range, flag_save=True)
    aaa = 1
    
    
def save_semseg_attr_img(save_dir, attrs, img, sem_seg_info, figsize):
    suffix_name = "{}-{}"
    seg_info = sem_seg_info.extract_seg_info(attrs)
    ori_seg_alpha = 0.8
    color_mode2 = cv2.COLOR_RGBA2BGRA
    
    attr_range = (np.min(attrs), np.max(attrs))
    
    ori_seg_save_name = 'ori-seg-' + suffix_name
    ori_segattr_save_name = 'attrseg-' + suffix_name
    ori_cattr_save_name = 'ori-colorattr-' + suffix_name
    
    for i, (i_label, seg_attrs_v) in enumerate(seg_info.items()):
        label_name = seg_attrs_v['name']
        label_color = seg_attrs_v['color']
        mask_attr = seg_attrs_v['mask_attr']
        mask = seg_attrs_v['mask']
        
        # 保存原始图和 seg 结合的图
        ori_seg_image = np.copy(img)
        ori_seg_image[mask] = (1 - ori_seg_alpha) * ori_seg_image[mask] + ori_seg_alpha * np.append(label_color, 255)
        utils_save.save_np_rgb_img(save_dir, ori_seg_save_name.format(i_label, label_name), ori_seg_image)
        
        # 保存原始图和 masked 归因的热力图
        utils_save.plot(save_dir, ori_segattr_save_name.format(i_label, label_name),
                        mask_attr, img, alpha=0.15, figsize=figsize, hm_range=attr_range, flag_save=True)
        
        # # 保存原始图和 masked 归因的 上色的结合图
        # binary_attr_mask = mask_attr > 0
        # ori_colorattr_image = np.copy(img)
        # ori_colorattr_image[binary_attr_mask] = (1 - alpha) * ori_colorattr_image[binary_attr_mask] + \
        #                                         alpha * np.append(label_color, 255)
        # ori_cattr_save_path = save_dir + '/' + ori_cattr_save_name.format(i_label, label_name)
        # cv2.imwrite(ori_cattr_save_path, img_proc.to_bgr(ori_colorattr_image, color_mode2))
    aaa = 1
    print("Save semantic seg images finished!")


def save_attr_np(save_dir, attr_index_str, attr_res, attr_res_blob):
    attr_res_save_name = "attr_res.npy"
    attr_res_blob_save_name = "attr_res_blob.npy"
    np.save(save_dir + '/' + attr_res_save_name, attr_res)
    np.save(save_dir + '/' + attr_res_blob_save_name, attr_res_blob)
    

def load_attr(save_dir):
    attr_res_save_name = "attr_res.npy"
    attr_res_blob_save_name = "attr_res_blob.npy"
    attr_res = np.load(save_dir + '/' + attr_res_save_name, allow_pickle=True)
    attr_res_blob = np.load(save_dir + '/' + attr_res_blob_save_name, allow_pickle=True)
    
    attr_res_res = {}
    attr_res_res['attrs_ori'] = attr_res.item().get('attrs_ori')
    attr_res_res['attrs_filtered'] = attr_res.item().get('attrs_filtered')
    attr_res_res['attr_index_str'] = attr_res.item().get('attr_index_str')
    
    attr_res_blob_res = {}
    attr_res_blob_res['xs'] = attr_res_blob.item().get('xs')
    attr_res_blob_res['ys'] = attr_res_blob.item().get('ys')
    attr_res_blob_res['sizes'] = attr_res_blob.item().get('sizes')
    attr_res_blob_res['covs'] = attr_res_blob.item().get('covs')
    attr_res_blob_res['features'] = attr_res_blob.item().get('features')
    attr_res_blob_res['spatial_style'] = attr_res_blob.item().get('spatial_style')
    attr_res_blob_res['features_sum'] = attr_res_blob.item().get('features_sum')
    attr_res_blob_res['spatial_style_sum'] = attr_res_blob.item().get('spatial_style_sum')
    # print(attr_res.item().keys())
    # print(attr_res.item().get('attrs_ori'))
    return attr_res_res, attr_res_blob_res


def save_attr_blob_np(save_dir, save_name, attr_res_blob):
    np.save(save_dir + '/' + save_name, attr_res_blob)
    
    
def load_attr_blob_np(save_dir, save_name):
    attr_res_blob = np.load(save_dir + '/' + save_name, allow_pickle=True)
    attr_res_blob_res = {}
    attr_res_blob_res['xs'] = attr_res_blob.item().get('xs')
    attr_res_blob_res['ys'] = attr_res_blob.item().get('ys')
    attr_res_blob_res['sizes'] = attr_res_blob.item().get('sizes')
    attr_res_blob_res['covs'] = attr_res_blob.item().get('covs')
    attr_res_blob_res['features'] = attr_res_blob.item().get('features')
    attr_res_blob_res['spatial_style'] = attr_res_blob.item().get('spatial_style')
    attr_res_blob_res['features_sum'] = attr_res_blob.item().get('features_sum')
    attr_res_blob_res['spatial_style_sum'] = attr_res_blob.item().get('spatial_style_sum')
    return attr_res_blob_res
    

def cal_save_intersect(save_dir, img_re, img_re_wo_blobs):
    intersect_masks = []
    save_name_general = 'gray-diff-{}'
    save_name1_general = 'ori-mask-wo-{}'
    save_name2_general = 'mask-wo-{}'
    
    # # 定义膨胀核（这里使用3x3的正方形核）
    # kernel = np.ones((3, 3), np.uint8)
    # 定义四联通的膨胀核
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    img_re = img_re.astype(np.float64)
    
    for i_re_img in range(40):
        i_blob_str = "{:02d}".format(i_re_img+1)
        save_name = save_name_general.format(i_blob_str)
        save_name1 = save_name1_general.format(i_blob_str)
        save_name2 = save_name2_general.format(i_blob_str)
        
        img_re_wo_blob = img_re_wo_blobs[i_re_img]
        # 计算两张图像之间的差异

        img_re_wo_blob = img_re_wo_blob.astype(np.float64)
        diff = np.max(np.abs(img_re - img_re_wo_blob), -1)
        
        # # 使用单一阈值分割 - mask 大
        # mask = np.zeros_like(diff)
        # mask[diff > 12] = 1
        
        # # # 使用单一阈值分割 - mask 中
        mask = np.zeros_like(diff)
        mask[diff > 42] = 1
        
        # # 使用Otsu阈值分割将汽车与背景分割开 - mask 小
        # # blur = cv2.GaussianBlur(np.uint8(diff), (3, 3), 0)
        # ret, mask = cv2.threshold(np.uint8(diff), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # mask = np.asarray(mask/255, np.float64)
        
        utils_save.save_np_gray_img(save_dir, save_name, diff)
        # utils_save.save_np_01_img(save_dir, save_name1, mask)
        
        # # 只使用最大联通域的策略
        # if np.max(mask) < 0.01:
        #     utils_save.save_np_01_img(save_dir, save_name2, mask)
        #     intersect_masks.append(mask)
        # else:
        #     # 进行连通组件分析
        #     _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        #
        #     # 找到最大连通区域的索引
        #     largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        #
        #     # 创建只包含最大连通区域的掩码
        #     largest_component_mask = np.zeros_like(mask)
        #     largest_component_mask[labels == largest_component_index] = 1
        #
        #     # 对最大连通区域掩码进行膨胀处理
        #     dilated_mask = cv2.dilate(largest_component_mask, kernel, iterations=2)
        #
        #     # 把膨胀后的, 单一联通的遮罩保存起来
        #     utils_save.save_np_01_img(save_dir, save_name2, dilated_mask)
        #     intersect_masks.append(dilated_mask)
        
        # 先腐蚀再膨胀的策略
        # 对最大连通区域掩码进行膨胀处理
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)
        if np.max(dilated_mask) < 0.01:
            utils_save.save_np_01_img(save_dir, save_name2, mask)
            intersect_masks.append(mask)
        else:
            # 进行连通组件分析
            _, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_mask.astype(np.uint8))
            # 找到最大连通区域的索引
            largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

            # 创建只包含最大连通区域的掩码
            largest_component_mask = np.zeros_like(mask)
            largest_component_mask[labels == largest_component_index] = 1
            
            # 把膨胀后的, 单一联通的遮罩保存起来
            utils_save.save_np_01_img(save_dir, save_name2, largest_component_mask)
            intersect_masks.append(largest_component_mask)
            
    intersect_masks = np.stack(intersect_masks)
    return intersect_masks


def cal_intersect_region_inv(save_dir, save_name_mask, attr_method, img_re, img_re_wo_blobs):
    """
    img_re_wo_blobs 是去掉了重要 blob 的结果
    img_re 是原始图像
    这个函数首先计算它们的差, 并形成一大片联通域
    之后利用图像中其他区域, 形成 mask, 也就是不重要区域为 1 的 mask
    """
    # # 定义膨胀核（这里使用3x3的正方形核）
    # kernel = np.ones((5, 5), np.uint8)
    # # 定义四联通的膨胀核
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    img_re = img_re.astype(np.float64)
    img_re_wo_blob = img_re_wo_blobs.astype(np.float64)
    
    # 计算两张图像之间的差异
    diff = np.max(np.abs(img_re - img_re_wo_blob), -1)
    
    # # 使用单一阈值分割 - mask 大
    # mask = np.zeros_like(diff)
    # mask[diff > 12] = 1
    
    # # # 使用单一阈值分割 - mask 中
    mask = np.zeros_like(diff)
    if attr_method == 'Saliency':
        mask[diff > 42] = 1
    else:
        mask[diff > 52] = 1
    
    # # 使用Otsu阈值分割将汽车与背景分割开 - mask 小
    # # blur = cv2.GaussianBlur(np.uint8(diff), (3, 3), 0)
    # ret, mask = cv2.threshold(np.uint8(diff), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # mask = np.asarray(mask/255, np.float64)
    
    # 先腐蚀再膨胀的策略
    # 对最大连通区域掩码进行膨胀处理
    if attr_method == 'Saliency':
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
    else:
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
        
    # # 进行连通组件分析
    # _, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_mask.astype(np.uint8))
    # # 找到最大连通区域的索引
    # largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    #
    # # 创建只包含最大连通区域的掩码
    # largest_component_mask = np.zeros_like(mask)
    # largest_component_mask[labels == largest_component_index] = 1
    
    # mask = inv_mask(largest_component_mask)
    mask = inv_mask(eroded_mask)
    return mask, diff


def cal_safe_res(img_ori, img_inv, img_mask, save_dir, img_name):
    """
    用正常方法生成反演, 计算反演与原始图像的差异,
    并形成最大联通域, 保存改变 mask,
    mask 周围模糊, 用 alpha-blend 混合图像, 形成 safe 论文 结果
    
    """
    # 定义保存的名字
    print("Curr img {}".format(img_name))
    save_name_mask = img_name + '-3mask'
    save_name_safe = img_name + '-3cf'
    save_name_hist = img_name + '-3hist'
    
    # 直方图均衡化
    # img_inv = match_histograms(img_inv, img_ori)
    # utils_save.save_np_rgb_img(save_dir, save_name_hist, img_inv)
    
    img_mask_max = np.max(img_mask, -1)
    mask_temp = np.zeros_like(img_mask_max)
    mask_temp[img_mask_max == 255] = 255
    
    # 进行连通组件分析
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_temp.astype(np.uint8))
    # 找到最大连通区域的索引
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # 创建只包含最大连通区域的掩码
    mask_mid = np.zeros_like(mask_temp)
    mask_mid[labels == largest_component_index] = 255
    mask_mid = mask_mid.astype(np.uint8)
    
    kernel_size = (45, 45)
    kernel = np.ones(kernel_size, np.uint8)
    
    # 对mask进行膨胀处理
    mask_mid = cv2.dilate(mask_mid, kernel, iterations=1)

    # 使用高斯模糊平滑边缘
    blurred_mask = cv2.GaussianBlur(mask_mid, (31, 31), 0)
    
    # 使用形态学变换进一步平滑边缘
    kernel = np.ones((25, 25), np.uint8)
    smoothed_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    
    # 确保mask保持二值化
    _, smoothed_mask = cv2.threshold(smoothed_mask, 240, 255, cv2.THRESH_BINARY)
    
    # 对平滑后的mask边缘进行模糊处理
    # mask = cv2.GaussianBlur(smoothed_mask, (61, 61), 0)
    mask = cv2.GaussianBlur(smoothed_mask, (1, 1), 0)
    # utils_save.save_np_gray_img(save_dir, save_name_mask, mask)
    
    # # 将 img_inv 和 img_ori 进行 alpha blend 融合
    # alpha = blurred_edges_mask / 255.
    # alpha = alpha[..., np.newaxis]
    # result = img_inv * alpha + img_ori * (1 - alpha)
    
    # 找到mask的中心点
    # mask_center = (mask.shape[1] // 2, mask.shape[0] // 2)
    mask_center = img_proc.get_bounding_rect_center(mask)
    # 使用无缝克隆进行图像拼接
    result = cv2.seamlessClone(img_inv, img_ori, mask, mask_center, cv2.NORMAL_CLONE)

    # 保存结果
    utils_save.save_np_rgb_img(save_dir, save_name_safe, result)
    
    aaa = 1

    
def inv_mask(ori_mask):
    # 将0变成1，1变成0
    mask = np.where(ori_mask == 0, 1, np.where(ori_mask == 1, 0, ori_mask))
    return mask
    

def cal_unimportant_region_intersect(save_dir, img_re, img_re_wo_blobs):
    save_name_general = 'gray-diff'
    save_name1_general = 'ori-mask-wo'
    save_name2_general = 'mask-wo'
    
    # # 定义膨胀核（这里使用3x3的正方形核）
    # kernel = np.ones((3, 3), np.uint8)
    # 定义四联通的膨胀核
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # 计算两张图像之间的差异
    img_re = img_re.astype(np.float64)
    img_re_wo_blob = img_re_wo_blobs.astype(np.float64)
    diff = np.max(np.abs(img_re - img_re_wo_blob), -1)
    
    # # 使用单一阈值分割 - mask 大
    # mask = np.zeros_like(diff)
    # mask[diff > 12] = 1
    
    # # # 使用单一阈值分割 - mask 中
    mask = np.zeros_like(diff)
    mask[diff > 42] = 1
    
    # # 使用Otsu阈值分割将汽车与背景分割开 - mask 小
    # # blur = cv2.GaussianBlur(np.uint8(diff), (3, 3), 0)
    # ret, mask = cv2.threshold(np.uint8(diff), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # mask = np.asarray(mask/255, np.float64)
    
    utils_save.save_np_gray_img(save_dir, save_name_general, diff)
    return mask


def save_masked_attr_hm(save_dir, attr_res, intersect_masks, img_re, figsize):
    save_name_general = 'attr-' + str(attr_res['attr_index_str']) + '-b{}'
    attrs_ori = attr_res['attrs_ori'].copy()
    attrs_filtered = attr_res['attrs_filtered'].copy()
    attr_range = (np.min(attrs_filtered), np.max(attrs_filtered))
    
    for i_blob in range(40):
        i_blob_str = "{:02d}".format(i_blob+1)
        save_name = save_name_general.format(i_blob_str)
        
        intersect_mask = intersect_masks[i_blob]
        attrs_ori[intersect_mask == 1] = 0
        attr_masked = attrs_filtered * intersect_mask
        save_masked_attr_img_hm(save_dir, save_name, attr_masked, img_re,
                                figsize, attr_range=attr_range)
    print("Save masked attr finished")


def save_masked_attr_ratio(save_dir, attr_res, intersect_masks):
    save_name = 'ratio_masked_attr.txt'
    attrs_ori = attr_res['attrs_ori'].copy()
    attrs_filtered = attr_res['attrs_filtered'].copy()
    
    attr_masked_arr = []
    for i_blob in range(40):
        intersect_mask = intersect_masks[i_blob]
        attr_masked = attrs_filtered * intersect_mask
        single_attr = np.sum(attr_masked)
        attr_masked_arr.append(single_attr)
    attr_masked_arr = np.stack(attr_masked_arr)
    attr_sum = np.sum(attr_masked_arr)
    
    attr_masked_sort, attr_masked_idx = utils_save.sort_np_arr(attr_masked_arr)
    attr_masked_sort_ratio = attr_masked_sort / attr_sum * 100
    # 设置保存到txt文件的格式化字符串
    fmt = "%-5d %.2f   %.2f"

    # 将NumPy数组和索引数组保存到txt文件
    np.savetxt(save_dir+'/'+save_name,
               np.column_stack((attr_masked_idx+1, attr_masked_sort_ratio, attr_masked_sort)),
               fmt=fmt)
    aaa = 1


def sort_blob_attr(attr_res_blob, opt):
    attr_size = attr_res_blob['sizes'][1:]
    # attr_spat = np.sum(np.abs(attr_res_blob['spatial_style']), -1)[1:]
    # attr_feat = np.sum(np.abs(attr_res_blob['features']), -1)[1:]
    attr_spat_sum = attr_res_blob['spatial_style_sum'][1:]
    attr_feat_sum = attr_res_blob['features_sum'][1:]
    
    attr_size_sort, attr_size_sort_idx = utils_save.sort_np_arr(attr_size)
    # attr_spat_sort, attr_spat_sort_idx = utils_save.sort_np_arr(attr_spat)
    # attr_feat_sort, attr_feat_sort_idx = utils_save.sort_np_arr(attr_feat)
    attr_spat_sum_sort, attr_spat_sum_sort_idx = utils_save.sort_np_arr(attr_spat_sum)
    attr_feat_sum_sort, attr_feat_sum_sort_idx = utils_save.sort_np_arr(attr_feat_sum)
    
    # 值越大, 留下的 blob 越多
    inter_len = get_cum_argmax_value(attr_feat_sum_sort, opt.attr_thres)
        
    # inter_len = get_cum_argmax_value(attr_feat_sum_sort, 0.55)
    intersection = np.intersect1d(np.intersect1d(attr_size_sort_idx[:inter_len],
                                  attr_spat_sum_sort_idx[:inter_len]),
                                  attr_feat_sum_sort_idx[:inter_len])
    # intersection = attr_spat_sum_sort_idx[:6]
    # print("attr_size_sort_idx", attr_size_sort_idx+1)
    # print("attr_spat_sort_idx", attr_spat_sort_idx+1)
    # print("attr_feat_sort_idx", attr_feat_sort_idx+1)
    part_blob_attrs = attr_feat_sum[intersection]
    return attr_feat_sum_sort, attr_feat_sum_sort_idx, part_blob_attrs, intersection


def sort_max_blob_attr(attr_res_blob, opt):
    attr_size = attr_res_blob['sizes'][1:]
    # attr_spat = np.sum(np.abs(attr_res_blob['spatial_style']), -1)[1:]
    # attr_feat = np.sum(np.abs(attr_res_blob['features']), -1)[1:]
    attr_spat_sum = attr_res_blob['spatial_style_sum'][1:]
    attr_feat_sum = attr_res_blob['features_sum'][1:]
    
    attr_size_sort, attr_size_sort_idx = utils_save.sort_np_arr(attr_size)
    attr_spat_sum_sort, attr_spat_sum_sort_idx = utils_save.sort_np_arr(attr_spat_sum)
    attr_feat_sum_sort, attr_feat_sum_sort_idx = utils_save.sort_np_arr(attr_feat_sum)
    
    # 值越大, 留下的 blob 越多
    inter_len = get_cum_argmax_value(attr_size_sort, opt.attr_thres)
    
    # inter_len = get_cum_argmax_value(attr_feat_sum_sort, 0.55)
    intersection = attr_size_sort_idx[:1]
    # intersection = attr_spat_sum_sort_idx[:6]
    # print("attr_size_sort_idx", attr_size_sort_idx+1)
    # print("attr_spat_sort_idx", attr_spat_sort_idx+1)
    # print("attr_feat_sort_idx", attr_feat_sort_idx+1)
    part_blob_attrs = attr_feat_sum[intersection]
    return attr_size_sort, attr_size_sort_idx, part_blob_attrs, intersection


def sort_blob_attr_has_minus(attr_res_blob, attr_thres):
    # blob attr 有负数的情况
    attr_size = attr_res_blob['sizes'][1:]
    # attr_spat = np.sum(np.abs(attr_res_blob['spatial_style']), -1)[1:]
    # attr_feat = np.sum(np.abs(attr_res_blob['features']), -1)[1:]
    attr_spat_sum = attr_res_blob['spatial_style_sum'][1:]
    attr_feat_sum = attr_res_blob['features_sum'][1:]
    
    attr_size = clear_minus_val(attr_size)
    attr_spat_sum = clear_minus_val(attr_spat_sum)
    attr_feat_sum = clear_minus_val(attr_feat_sum)
    
    attr_size_sort, attr_size_sort_idx = utils_save.sort_np_arr(attr_size)
    # attr_spat_sort, attr_spat_sort_idx = utils_save.sort_np_arr(attr_spat)
    # attr_feat_sort, attr_feat_sort_idx = utils_save.sort_np_arr(attr_feat)
    attr_spat_sum_sort, attr_spat_sum_sort_idx = utils_save.sort_np_arr(attr_spat_sum)
    attr_feat_sum_sort, attr_feat_sum_sort_idx = utils_save.sort_np_arr(attr_feat_sum)
    
    # 值越大, 留下的 blob 越多
    inter_len = get_cum_argmax_value(attr_feat_sum_sort, attr_thres)
    if inter_len < 5:
        inter_len = 5
    else:
        inter_len = 6
        
    intersection = attr_spat_sum_sort_idx[:inter_len]
    # print("attr_size_sort_idx", attr_size_sort_idx+1)
    # print("attr_spat_sort_idx", attr_spat_sort_idx+1)
    # print("attr_feat_sort_idx", attr_feat_sort_idx+1)
    part_blob_attrs = attr_feat_sum[intersection]
    return attr_feat_sum_sort, attr_feat_sum_sort_idx, part_blob_attrs, intersection


def sort_unimportant_blob_attr(attr_res_blob, inter_len):
    attr_size = attr_res_blob['sizes'][1:]
    # attr_spat = np.sum(np.abs(attr_res_blob['spatial_style']), -1)[1:]
    # attr_feat = np.sum(np.abs(attr_res_blob['features']), -1)[1:]
    attr_spat_sum = attr_res_blob['spatial_style_sum'][1:]
    attr_feat_sum = attr_res_blob['features_sum'][1:]
    
    attr_size_sort, attr_size_sort_idx = utils_save.sort_np_arr(attr_size)
    # attr_spat_sort, attr_spat_sort_idx = utils_save.sort_np_arr(attr_spat)
    # attr_feat_sort, attr_feat_sort_idx = utils_save.sort_np_arr(attr_feat)
    attr_spat_sum_sort, attr_spat_sum_sort_idx = utils_save.sort_np_arr(attr_spat_sum)
    attr_feat_sum_sort, attr_feat_sum_sort_idx = utils_save.sort_np_arr(attr_feat_sum)
    
    # 值越大, 留下的 blob 越多
    intersection = attr_spat_sum_sort_idx[-inter_len:]
    part_blob_attrs = attr_feat_sum[intersection]
    return attr_feat_sum_sort, attr_feat_sum_sort_idx, part_blob_attrs, intersection


def clear_minus_val(arr):
    arr = arr * (arr > 0)
    return arr


def get_cum_argmax_value(arr, q):
    # 我有一个1D的np数组,里面数值是从大到小的,我想找到一个位置,在这个位置之前的数值之和占np数组sum值的70%
    # q 0-1
    arr_sum = np.sum(arr)
    # 计算累积和
    cumulative_sum = np.cumsum(arr)
    threshold = q * arr_sum
    index = np.argmax(cumulative_sum >= threshold)
    return index
    

def cal_save_blob_attr_ratio(save_dir, attr_res_blob):
    # 给 blob attr 排序, 并保存到 txt
    save_name = "ratio_blob_attr.txt"
    
    attr_feat_sort, attr_feat_sort_idx, _ = sort_blob_attr(attr_res_blob)
    attr_feat_sort_ratio = attr_feat_sort / np.sum(attr_feat_sort) * 100
    
    # 设置保存到txt文件的格式化字符串
    fmt = "%-5d %.2f   %.2f"

    # 将NumPy数组和索引数组保存到txt文件
    np.savetxt(save_dir+'/'+save_name,
               np.column_stack((attr_feat_sort_idx+1, attr_feat_sort_ratio, attr_feat_sort)),
               fmt=fmt)
    aaa = 1
