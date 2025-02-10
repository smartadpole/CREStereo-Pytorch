from collections import OrderedDict
from nets import Model
from utils import inference
import argparse
import torch
from file import MkdirSimple, match_images
import os
from tqdm.contrib import tzip
from PIL import Image
import numpy as np
from time import time
import cv2
from file import Walk, MkdirSimple

DATA_TYPE = ['kitti', 'dl', 'depth', 'server']

DIR_SHAPE = [
    ['left', 'right'],
    ['image_02', 'image_03'],
    ['cam0', 'cam1'],
    ['L', 'R']
]


def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--ignore_target', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--model_file', type=str, default='models/crestereo_eth3d.pth')
    parser.add_argument('--output', type=str)
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")
    parser.add_argument('--max_depth', type=int, default=1000, help="the valide max depth (cm)")
    parser.add_argument('--scale', type=float, default=1, help="scale image to super resolution")

    args = parser.parse_args()

    return args



def GetImages(path):
    matched_lists = None
    for l, r in DIR_SHAPE:
        left = os.path.join(path, l)
        right = os.path.join(path, r)
        if os.path.exists(left) and os.path.exists(right):
            matched_lists = match_images([left, right])
            break

    return matched_lists[0], matched_lists[1], len(left.rstrip('/'))


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


def WriteDepth(predict_np, limg, path, name, bf, max_value=65535):
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
    if bf > 0:
        depth_img = bf / predict_np * 100  # to cm
        depth_img = np.clip(depth_img, 0, max_value)
        depth_img_u16 = depth_img / max_value * 65535
        depth_img_u16 = depth_img_u16.astype("uint16")
    else:
        depth_img_u16 = predict_np
        depth_img = depth_img_u16 / 65535 * 2500

    cv2.imwrite(output_gray, depth_img_u16)
    # return


    depth_norm = np.clip(depth_img, 1, max_value)
    depth_norm = 1.0 / depth_norm
    depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)

    if bf < 0:
        predict_np = depth_norm

    color_img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    limg_cv = limg # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    # todo: hao 2025-02-10 14:13 - if input depth image
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


def main():
    args = GetArgs()
    bf = args.bf * args.scale

    # output_dir = os.path.join(args.output, os.path.basename(args.data_path))
    output_dir = os.path.join(args.output, 'filter')
    output_dir_left = os.path.join(args.output, "left")
    output_dir_right = os.path.join(args.output, "right")

    if not args.no_cuda:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    match_files = GetImages(args.data_path)
    if not match_files:
        print("No matched files in the path: ", args.data_path)
        exit(0)

    left_files, right_files, root_len = match_files

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

    start = 0
    left_files = left_files[start:]
    right_files = right_files[start:]

    for left_image_file, right_image_file in tzip(left_files, right_files):
        if not os.path.exists(left_image_file) or not os.path.exists(right_image_file):
            continue

        output_name = left_image_file[root_len + 1:]

        # todo: hao 2025-02-08 15:25 - update the logic
        if args.ignore_target:
            name = os.path.splitext(output_name)[0] + ".png"
            output_concat_color = os.path.join(output_dir, "concat_color", name)
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

        left_flip = cv2.flip(left_img, 1)
        right_flip = cv2.flip(right_img, 1)
        dispR_filp = inference(right_flip, left_flip, model, n_iter=20)
        disp_right = cv2.flip(dispR_filp, 1)

        # clip depth,
        # remove too large distance

        # scale to origin size（if scale<1）
        if abs(args.scale - 1) > 1e-6:
            disp_left = cv2.resize(disp_left, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
            disp_right = cv2.resize(disp_right, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
            left_img_show = cv2.resize(left_img, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
            right_img_show = cv2.resize(right_img, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
        else:
            left_img_show = left_img
            right_img_show = right_img

        WriteDepth(disp_left, left_img_show, output_dir_left, output_name, bf, args.max_depth)
        WriteDepth(disp_right, right_img_show, output_dir_right, output_name, bf, args.max_depth)

if __name__ == '__main__':
    main()
