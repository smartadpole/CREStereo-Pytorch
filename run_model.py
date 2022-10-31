import torch
import numpy as np
import cv2
import os
import argparse
import scipy.io as io

from nets import Model
from utils import inference

device = 'cuda'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input_folder', help='Contains subfolders cam0/cam1', required=True)
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

    model_path = "models/crestereo_eth3d.pth"

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()

    # pred = inference(imgL, imgR, model, n_iter=5)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    #     with record_function("model_inference"):
    #         pred = inference(imgL, imgR, model, n_iter=5)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # prof.export_chrome_trace("../trace.json")
    # prof.export_stacks("../profiler_stacks.txt", "self_cuda_time_total")
    # print(model)

    pred = inference(imgL, imgR, model, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

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
