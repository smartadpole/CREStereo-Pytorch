import torch
import numpy as np
import cv2
import os
import argparse
from collections import OrderedDict

import scipy.io as io
import matplotlib.pyplot as plt
from nets import Model
from utils import inference, evaluate_compute

device = 'cpu'


def convert_depth_to_display(disp: np.ndarray, t: float = 1.):
    """

    """
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-l', '--left_image_fn', help='Filename of left image', required=True)
    parser.add_argument('-r', '--right_image_fn', help='Filename of left image', required=True)
    parser.add_argument('-o', '--output_directory', help="Directory to save output", default="demo_output")
    parser.add_argument('-e', action='store_true', help="Use to compute params and MACs needed by network")
    args = parser.parse_args()

    left_img = cv2.imread(args.left_image_fn)
    right_img = cv2.imread(args.right_image_fn)

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

    model_path = './train_log/models/latest.pth'  # "models/crestereo_eth3d.pth"
    state_dict = torch.load(model_path)
    model_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[7:]  # remove `module.`
        model_state_dict[name] = v

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)
    model.eval()

    # evaluate_compute(imgL.copy(), imgR.copy(), model)
    # pred_eval = inference(imgL.copy(), imgR.copy(), model, n_iter=4)
    pred = inference(imgL, imgR, model, n_iter=20, device=device)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    disp_vis = convert_depth_to_display(disp)
    #disp_vis_eval = convert_depth_to_display(cv2.resize(pred_eval, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # axes[0].imshow(1. / pred)
    # axes[0].set_title("n-iter=20")
    # axes[1].imshow(1. / pred_eval)
    # axes[1].set_title("n-iter=4")

    combined_img = np.hstack((left_img, disp_vis))

    output_directory = args.output_directory
    filename = os.path.basename(args.left_image_fn)
    filename_wo_ext = os.path.splitext(filename)[0]

    output_filename = os.path.join(output_directory, f"{filename_wo_ext}.png")
    print(f"output_filename: {output_filename}")
    os.makedirs(output_directory, exist_ok=True)
    cv2.imwrite(output_filename, disp_vis)
    io.savemat(os.path.join(output_directory, f"{filename_wo_ext}.mat"), dict(disp=disp))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", combined_img)
    cv2.waitKey(10)

    plt.figure()
    plt.imshow(1. / pred)
    plt.show()
