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
from tqdm.contrib import tzip

DATA_TYPE = ['kitti', 'dl', 'depth', 'server']


def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--ignore_target', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--model_file', type=str, default='models/crestereo_eth3d.pth')
    parser.add_argument('--file_name', help='image list in file', required=True)
    parser.add_argument('--bf', type=float, default=14.2, help="baseline length multiply focal length")

    args = parser.parse_args()

    return args


def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        paths = Walk(path, ['jpg', 'png', 'jpeg'])
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

def gray_scale_region(gray_img, min_value=1, max_value=255):
    min_valid_index = gray_img > min_value

    max_valid_index = gray_img < max_value

    if (not (min_valid_index.any())):
        gray_img[:,:] = min_value
        return gray_img
    if (not (max_valid_index.any())):
        gray_img[:, :] = max_value
        return gray_img

    min_scale_value = np.min(gray_img[min_valid_index])

    max_scale_value = np.max(gray_img[max_valid_index])

    scale_gray = gray_img.copy()
    scale_gray[gray_img < min_value] = min_scale_value
    scale_gray[gray_img > max_value] = max_scale_value
    scale_gray = 1.0 / scale_gray
    scale_gray = (scale_gray - np.min(scale_gray))/(np.max(scale_gray) - np.min(scale_gray)) * 255
    return scale_gray

def get_file_name(file_name, image_path):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    image_list = []
    depth_list = []
    for image_name in file_lists:
        image_full_path = os.path.join(image_path, image_name)
        image_dest_path = image_full_path.replace("REMAP", "DEPTH/CREStereo").split()[0]
        image_dest_path = image_dest_path.replace(".jpg", ".png")
        image_list.append(image_full_path)
        depth_list.append(image_dest_path)

    return image_list, depth_list

def write_by_img_list(img_list, depth_list, model):
    for left_image_file, depth_image_file in tzip(img_list, depth_list):
        if "cam0/" in left_image_file:
            right_image_file = left_image_file.replace("cam0/", "cam1/")
        elif "cam1/" in left_image_file:
            right_image_file = left_image_file
            left_image_file = right_image_file.repalce("cam1/", "cam0/")
            print("right_image_file: ", right_image_file)
        else:
            print("'cam0/' or 'cam1/' not in image name, continue")
            continue
        left_image_file = left_image_file.split()[0]
        right_image_file = right_image_file.split()[0]

        if not os.path.exists(left_image_file) or not os.path.exists(right_image_file):
            print("image in cam0 or in cam1 not exist, continue")
            assert 0
            continue

        left_img = cv2.imread(left_image_file)
        right_img = cv2.imread(right_image_file)


        in_h, in_w = left_img.shape[:2]

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

        in_h, in_w = left_img.shape[:2]
        # Resize image in case the GPU memory overflows
        eval_h, eval_w = (in_h, in_w)
        assert eval_h % 8 == 0, "input height should be divisible by 8"
        assert eval_w % 8 == 0, "input width should be divisible by 8"

        imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

        with torch.no_grad():
            start = time()
            predict_np = inference(imgL, imgR, model, n_iter=20)
            # print("use: ", (time() - start))

        MkdirSimple(depth_image_file)

        predict_np_scale = gray_scale_region(predict_np, 5, 240)
        if (not predict_np_scale.any()):
            print(left_image_file)
            continue

        cv2.imwrite(depth_image_file, predict_np_scale)

def main():
    args = GetArgs()

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

    img_list, depth_list = get_file_name(args.file_name, args.data_path)

    write_by_img_list(img_list, depth_list, model)

if __name__ == '__main__':
    main()
