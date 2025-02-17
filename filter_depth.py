#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: filter_depth.py
@time: 2025/2/10 13:52
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

from utils import inference
import argparse
import numpy as np
import cv2
from file import Walk, MkdirSimple, match_images
from test_image import WriteDepth
from tqdm.contrib import tzip

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument('--lr_threshold', type=float, default=-1, help="ignore the disp in left and right when diff ratio larger than lr_threshold; less than 0 means no filter")
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")
    parser.add_argument('--max_depth', type=int, default=1000, help="the valide max depth (cm)")

    args = parser.parse_args()
    return args


def large_region(img1, img2):
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    area_threshold = 7000
    num_labels1, labels1 = cv2.connectedComponents(img1, connectivity=8)
    num_labels2, labels2 = cv2.connectedComponents(img2, connectivity=8)

    areas2 = {}
    for label in range(1, num_labels2):
        areas2[label] = np.sum(labels2 == label)

    output = np.zeros_like(img1)

    # 遍历图1中每个连通域（标签从1开始，0为背景）
    for label1 in range(1, num_labels1):
        mask1 = (labels1 == label1)

        overlapping_labels = labels2[mask1]
        unique_labels = np.unique(overlapping_labels)
        unique_labels = unique_labels[unique_labels != 0]

        keep = False
        for l in unique_labels:
            if areas2.get(l, 0) > area_threshold:
                print(areas2.get(l, 0))
                keep = True
                break

        if keep:
            output[mask1] = 255

    return output

import numpy as np

def fill_holes_simple(disp):
    """
    use 4-neighbors to fill holes in disp

    disp: shape(H,W)，0 is invalid pixel to fill
    return: disp_filled
    """
    disp_filled = disp.copy()
    H, W = disp_filled.shape

    # --- left→right
    for y in range(H):
        for x in range(1, W):
            if disp_filled[y, x] == 0:
                disp_filled[y, x] = disp_filled[y, x-1]

    # ---  right→left
    for y in range(H):
        for x in range(W-2, -1, -1):
            if disp_filled[y, x] == 0:
                disp_filled[y, x] = disp_filled[y, x+1]

    # --- up→down
    for x in range(W):
        for y in range(1, H):
            if disp_filled[y, x] == 0:
                disp_filled[y, x] = disp_filled[y-1, x]

    # --- down→up
    for x in range(W):
        for y in range(H-2, -1, -1):
            if disp_filled[y, x] == 0:
                disp_filled[y, x] = disp_filled[y+1, x]

    return disp_filled


def left_right_consistency_check(dispL, dispR, alpha=0.1):
    """
    使用左右一致性检查过滤 dispL 中的无效像素：
    dispL[y,x] 和 dispR[y, x - dispL[y,x]] 应该一致（在阈值内）否则置0
    dispL, dispR: shape(h, w)
    threshold: 允许的视差差异
    return: dispL_filtered
    """
    if alpha < 0:
        return dispL

    h, w = dispL.shape

    x_coords = np.arange(w)
    x_r = (x_coords[None, :] - dispL).astype(int)
    occlusion_r = (x_r < 0) | (x_r >= w)
    x_r = np.clip(x_r, 0, w - 1)  # Ensure indices are within bounds
    valid = (dispL > 0)
    dispL_filtered = np.where(valid, dispL, 0)
    dispR_align = dispR[np.arange(h)[:, None], x_r]
    diff_l = np.abs(dispL_filtered - dispR_align)

    x_l = (x_coords[None, :] + dispR).astype(int)
    occlusion_l = (x_l < 0) | (x_l >= w)
    x_l = np.clip(x_l, 0, w - 1)  # Ensure indices are within bounds
    valid = (dispR > 0)
    dispR_filtered = np.where(valid, dispR, 0)
    dispL_align = dispL[np.arange(h)[:, None], x_l]
    diff_r = np.abs(dispR_filtered - dispL_align)
    diff_l_hat = diff_r[np.arange(h)[:, None], x_r]

    diff = np.minimum(diff_l, diff_l_hat)
    diff[occlusion_r] = diff_l[occlusion_r]

    edge_occlusion =  (diff_l / dispL_filtered) > 0.8
    edge_occlusion_dilate = cv2.dilate(edge_occlusion.astype(np.uint8), np.ones((10, 10), np.uint8))
    diff[edge_occlusion_dilate > 0] = 0
    diff_no_occlusion = fill_holes_simple(diff)


    threshold_map = alpha * dispL_filtered
    bad = (diff > threshold_map) & valid
    bad2 = (diff_no_occlusion > threshold_map) & valid
    bad2 = bad2 | bad

    # edge_occlusion_filter = large_region(edge_occlusion_dilate, bad)
    # edge_occlusion_filter = edge_occlusion_filter > 0
    # dispL_filtered[edge_occlusion_filter] = 0
    # bad = bad | edge_occlusion_filter

    bad = cv2.morphologyEx(bad2.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    bad = cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bad = bad > 0

    dispL_filtered[bad] = -1 # disp = -1 means invalid, the dpeth less than 0

    return dispL_filtered

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].imshow(dispL, cmap='gray')
    axs[0, 0].set_title('dispL')
    axs[0, 1].imshow(bad, cmap='gray')
    axs[0, 1].set_title('bad')
    axs[1, 0].imshow(diff_l, cmap='gray')
    axs[1, 0].set_title('diff_l')
    axs[1, 1].imshow(diff, cmap='gray')
    axs[1, 1].set_title('diff')
    axs[2, 0].imshow(dispL_filtered, cmap='gray')
    axs[2, 0].set_title('dispL_filtered')
    axs[2, 1].imshow(edge_occlusion, cmap='gray')
    axs[2, 1].set_title('edge_occlusion')

    plt.show()


    return dispL_filtered

def flip_consistency_check(disp, left_img, right_img, model, threshold=1.0, n_iter=20):
    """
    对输入图像水平翻转，再进行推理得到 disp_flip
    然后将 disp_flip 翻转回去，与 disp 对比，差异较大的地方置为0
    disp: 原始视差
    return: disp_flip_filtered
    """
    # 翻转左右图像
    left_flip = cv2.flip(left_img, 1)
    right_flip = cv2.flip(right_img, 1)
    # 推理得到 disp_flip
    disp_flip = inference(left_flip, right_flip, model, n_iter=n_iter)
    disp_flip = cv2.flip(disp_flip, 1).astype(np.float32)  # 再翻转回来

    # 对比差异
    diff = np.abs(disp - disp_flip)
    disp_max = np.minimum(disp, disp_flip)
    diff /= disp_max
    threshold_per = np.percentile(diff[diff > 0], 90)
    mask = diff <= min(threshold, threshold_per)
    disp_flip_filtered = np.where(mask, disp, 0)
    return disp_flip_filtered

def bilateral_filter_depth(depth, d=5, sigmaColor=2.0, sigmaSpace=5.0):
    """
    对有效区域(>0)执行双边滤波
    """
    depth_filtered = depth.copy().astype(np.float32)
    # OpenCV双边滤波会把0也当成参与滤波的值，需要先做一个mask保护无效像素
    valid_mask = (depth_filtered > 0).astype(np.uint8)
    # 为了让双边滤波只作用于有效区域，可以将无效区域临时替换成深度均值或其他策略
    valid_vals = depth_filtered[depth_filtered > 0]
    if len(valid_vals) > 0:
        mean_val = valid_vals.mean()
        depth_filtered[depth_filtered <= 0] = mean_val
    # 应用双边滤波
    depth_filtered = cv2.bilateralFilter(depth_filtered, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    # 滤波后再把原本无效区域恢复为0
    depth_filtered[valid_mask == 0] = 0
    return depth_filtered

def main():
    args = GetArgs()

    output_dir = os.path.join(args.data, 'filter')
    left_root = os.path.join(args.data, 'left', 'depth')
    right_root = os.path.join(args.data, 'right', 'depth')
    rgb_root = os.path.join(args.data, 'rgb', 'left')
    root_len_left = len(left_root.rstrip('/'))
    left_files, right_files, rgb_files = match_images([left_root, right_root, rgb_root])
    start = 0
    left_files = left_files[start:]
    right_files = right_files[start:]
    rgb_files = rgb_files[start:]

    for left_file, right_file, rgb_file in tzip(left_files, right_files, rgb_files):
        left_img = cv2.imread(left_file, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right_file, cv2.IMREAD_UNCHANGED)

        left_img =  65535.0 / left_img * args.bf * 100 / args.max_depth
        right_img =  65535.0 / right_img * args.bf * 100 / args.max_depth

        dispL_filtered = left_right_consistency_check(left_img, right_img, alpha=args.lr_threshold)
        name = left_file[root_len_left+1:]
        WriteDepth(dispL_filtered, rgb, output_dir, name, args.bf, args.max_depth)


if __name__ == '__main__':
    main()
