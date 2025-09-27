
import os
import cv2


def read_proc_img(image_path, color_mode=cv2.COLOR_BGR2RGB):
    """
    读取图像并转换为 BGR
    """
    # 读取图像,
    img = cv2.imread(image_path)
    if color_mode is not None and img is not None:
        img = cv2.cvtColor(img, color_mode)
    return img


def to_bgr(img, color_mode=cv2.COLOR_RGB2BGR):
    img = cv2.cvtColor(img, color_mode)
    return img


def get_cv2_img_size(img):
    height, width, channels = img.shape
    pixel_num = height * width
    return height, width, channels, pixel_num


def torch_norm(tensor):
    # make_grid 函数里面的一部分
    value_range = (-1, 1)
    tensor = tensor.clone()
    
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))
    norm_range(tensor, value_range)
    return tensor


def get_centroid(mask):
    # 计算图像矩
    moments = cv2.moments(mask)
    
    # 计算重心
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        cX, cY = mask.shape[1] // 2, mask.shape[0] // 2  # 作为备用的中心点
    
    return (cX, cY)


def get_bounding_rect_center(mask):
    # 查找非零像素点的外接矩形
    x, y, w, h = cv2.boundingRect(mask)
    
    # 计算外接矩形的中心点
    centerX = x + w // 2
    centerY = y + h // 2
    
    return (centerX, centerY)
