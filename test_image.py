from collections import OrderedDict
from nets import Model
from utils import inference
import argparse
import torch
from torchvision import transforms
import os
from tqdm.contrib import tzip
from PIL import Image
import numpy as np
from time import time
import cv2
from file import Walk, MkdirSimple

DATA_TYPE = ['kitti', 'dl', 'depth', 'server']


def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--ignore_target', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--model_file', type=str, default='models/crestereo_eth3d.pth')
    parser.add_argument('--output', type=str)
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")
    parser.add_argument('--max_depth', type=int, default=1000, help="the valide max depth")
    parser.add_argument('--scale', type=float, default=1, help="scale image to super resolution")

    args = parser.parse_args()

    return args


def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        if os.path.exists(os.path.join(path, 'all.txt')):
            paths = [os.path.join(path, l.strip('\n').strip()) for l in open(os.path.join(path, 'all.txt')).readlines()]
        else:
            paths = Walk(path, ['jpg', 'jpeg', 'png', 'bmp', 'pfm'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    left_files, right_files = [], []
    if 'kitti' == flag:
        left_files = [f for f in paths if 'image_02' in f]
        right_files = [f.replace('/image_02/', '/image_03/') for f in left_files]
    elif 'dl' == flag:
        left_files = [f for f in paths if 'cam0' in f]
        right_files = [f.replace('/cam0/', '/cam1/') for f in left_files]
    elif 'depth' == flag:
        left_files = [f for f in paths if 'left' in f and 'disp' not in f]
        right_files = [f.replace('left/', 'right/').replace('left.', 'right.') for f in left_files]
    elif 'server' == flag:
        left_files = [f for f in paths if '.L' in f]
        right_files = [f.replace('L/', 'R/').replace('L.', 'R.') for f in left_files]
    else:
        raise Exception("Do not support mode: {}".format(flag))

    return left_files, right_files, root_len


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)

    return depth_img_rgb.astype(np.uint8)


def WriteDepth(predict_np, limg, path, name, bf, max_value):
    name = os.path.splitext(name)[0] + ".png"
    name = name.replace('left', 'depth').replace('right', 'depth').replace('cam0', 'depth').replace('cam1', 'depth')
    output_concat_color = os.path.join(path, "rgb_color", name)
    output_concat_gray = os.path.join(path, "rgb_depth", name)
    output_concat_depth = os.path.join(path, "rgb_blue", name)
    output_concat = os.path.join(path, "concat", name)
    output_gray = os.path.join(path, "depth", name)
    output_float = os.path.join(path, "tiff", os.path.splitext(name)[0] + ".tiff")
    output_depth = os.path.join(path, "depth_blue", name)
    output_color = os.path.join(path, "color", name)

    MkdirSimple(output_gray)
    # MkdirSimple(output_float)
    depth_img = bf / predict_np * 100  # to cm

    # if depth_img.max() > max_value or depth_img.min() < 0:
    #     print("\nmax ", depth_img.max(), " min ", depth_img.min())
    #     print(name)
    # cv2.imwrite(output_float, depth_img)
    depth_img = np.clip(depth_img, 0, max_value)
    depth_img_u16 = depth_img / max_value * 65535
    depth_img_u16 = depth_img_u16.astype("uint16")

    cv2.imwrite(output_gray, depth_img_u16)
    # return

    depth_norm = np.clip(depth_img, 1, max_value)
    depth_norm = 1.0 / depth_norm
    depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)
    color_img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    limg_cv = limg # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_rgb = GetDepthImg(depth_img)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    # MkdirSimple(output_concat_color)
    # MkdirSimple(output_concat_gray)
    # MkdirSimple(output_concat_depth)
    # MkdirSimple(output_depth)
    # MkdirSimple(output_color)
    MkdirSimple(output_concat)

    # cv2.imwrite(output_concat_color, concat_img_color)
    # cv2.imwrite(output_concat_gray, concat_img_gray)
    # cv2.imwrite(output_color, color_img)
    # cv2.imwrite(output_depth, depth_img_rgb)
    # cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)

def left_right_consistency_check(dispL, dispR, threshold=1.0):
    """
    使用左右一致性检查过滤 dispL 中的无效像素：
    dispL[y,x] 和 dispR[y, x - dispL[y,x]] 应该一致（在阈值内）否则置0
    dispL, dispR: shape(h, w)
    threshold: 允许的视差差异
    return: dispL_filtered
    """
    h, w = dispL.shape
    dispL_filtered = dispL.copy()
    x_coords = np.arange(w)

    for y in range(h):
        d_row = dispL[y]
        d_row_r = dispR[y]
        # x' = x - d
        x_r = (x_coords - d_row).astype(int)
        # 有效性检查
        valid = (x_r >= 0) & (x_r < w) & (d_row > 0)
        # 对不在范围内的像素设为0
        dispL_filtered[y, ~valid] = 0
        idx = np.where(valid)[0]
        diff = np.abs(d_row[idx] - d_row_r[x_r[idx]])
        # set large difference to zero
        bad = diff > threshold
        dispL_filtered[y, idx[bad]] = 0

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
    bf = args.bf * args.scale

    output_directory = args.output

    if not args.no_cuda:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    left_files, right_files, root_len = [], [], []
    for k in DATA_TYPE:
        left_files, right_files, root_len = GetImages(args.data_path, k)
        if len(left_files) != 0:
            break

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    ckpt = torch.load(args.model_file)
    if 'optim_state_dict' in ckpt.keys():
        model_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k[7:]  # remove `module.`
            model_state_dict[name] = v
    else:
        model_state_dict = ckpt

    model.load_state_dict(model_state_dict, strict=True)

    if use_cuda:
        model.cuda()
    model.eval()

    # 主要新增参数
    lr_threshold = 1.0     # ignore the disp in left and right when larger than lr_threshold
    flip_threshold = 10.0   # 翻转一致性阈值
    do_flip_consistency = True  # 是否执行翻转一致性检查
    do_bilateral = True         # 是否执行双边滤波
    bilateral_params = dict(d=5, sigmaColor=2.0, sigmaSpace=5.0)

    for left_image_file, right_image_file in tzip(left_files, right_files):
        if not os.path.exists(left_image_file) or not os.path.exists(right_image_file):
            continue

        output_name = left_image_file[root_len + 1:]
        if args.ignore_target:
            name = os.path.splitext(output_name)[0] + ".png"
            output_concat_color = os.path.join(args.output, "concat_color", name)
            if os.path.exists(output_concat_color):
                continue

        left_img = cv2.imread(left_image_file)
        right_img = cv2.imread(right_image_file)

        org_h, org_w = left_img.shape[:2]
        in_h, in_w = left_img.shape[:2]
        scale_h, scale_w = [int(dim * args.scale) for dim in left_img.shape[:2]]
        left_img = cv2.resize(left_img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)

        if in_h % 8 != 0:
            pad_h = in_h % 8
            left_img = np.pad(left_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')
            right_img = np.pad(right_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')

        if in_w % 8 != 0:
            pad_w = in_w % 8
            left_img = np.pad(left_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')
            right_img = np.pad(right_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')

        in_h, in_w = left_img.shape[:2]

        if in_h % 8 != 0:
            pad_h = in_h % 8
            left_img = np.pad(left_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')
            right_img = np.pad(right_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')

        if in_w % 8 != 0:
            pad_w = in_w % 8
            left_img = np.pad(left_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')
            right_img = np.pad(right_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')

        eval_h, eval_w = left_img.shape[:2]
        assert eval_h % 8 == 0, "input height should be divisible by 8"
        assert eval_w % 8 == 0, "input width should be divisible by 8"

        imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

        with torch.no_grad():
            start = time()
            disp_left = inference(imgL, imgR, model, n_iter=20)
            print("inference use: {:.2f} ms".format((time() - start) * 1000))
            #
            # left_flip = cv2.flip(imgR, axis=1)
            # right_flip = cv2.flip(imgL, axis=1)
            # disp_right = inference(imgR, imgL, model, n_iter=20)

        # dispL_filtered = left_right_consistency_check(disp_left, disp_right, threshold=lr_threshold)
        dispL_filtered = disp_left

        # B. 翻转一致性检查 (可选)
        if do_flip_consistency:
            dispL_filtered = flip_consistency_check(dispL_filtered, left_img, right_img, model,
                                                        threshold=flip_threshold, n_iter=20)

        # # C. 可选：双边滤波增强 (仅对 >0 的有效区域)
        # if do_bilateral:
        #     disp_refined = bilateral_filter_depth(disp_flip_filtered,
        #                                           d=bilateral_params['d'],
        #                                           sigmaColor=bilateral_params['sigmaColor'],
        #                                           sigmaSpace=bilateral_params['sigmaSpace'])
        # else:
        #     disp_refined = disp_flip_filtered

        # 这里 disp_refined 就是处理后的视差图，直接替换原本 predict_np
        predict_np = dispL_filtered

        # clip depth,
        # remove too large distance

        # scale to origin size（if scale<1）
        if abs(args.scale - 1) > 1e-6:
            predict_np = cv2.resize(predict_np, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
            left_img_show = cv2.resize(left_img, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
        else:
            left_img_show = left_img

        WriteDepth(predict_np, left_img_show, args.output, output_name, bf, args.max_depth)

if __name__ == '__main__':
    main()
