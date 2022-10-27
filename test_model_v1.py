import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary
import numpy as np
import cv2
import os
import argparse 
import scipy.io as io
import thop
import json
from thop import clever_format
from imread_from_url import imread_from_url

from nets import Model

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=4, init_flow=True):

    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = torch.tensor(imgL.astype("float32")).to(device)
    imgR = torch.tensor(imgR.astype("float32")).to(device)
    
    if init_flow:
        imgL_dw2 = F.interpolate(
            imgL,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        imgR_dw2 = F.interpolate(
            imgR,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
    # print(imgR_dw2.shape)
    with torch.inference_mode():
        if init_flow:
            pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
            pred_flow = model(imgL, imgR, iters=n_iter // 2, flow_init=pred_flow_dw2)
        else:
            pred_flow = model(imgL, imgR, iters=n_iter, flow_init=None)            

    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp


def evaluate_compute(model, imgL:np.ndarray, imgR:np.ndarray):
    
    imgL_ = imgL.transpose(2, 0, 1)
    imgR_ = imgR.transpose(2, 0, 1)
    imgL_ = np.ascontiguousarray(imgL_[None, :, :, :])
    imgR_ = np.ascontiguousarray(imgR_[None, :, :, :])

    imgL_ = torch.tensor(imgL_.astype("float32")).to(device)
    imgR_ = torch.tensor(imgR_.astype("float32")).to(device)

    
    n_iters=8
    summary_depth=4
    imgL_dw2 = F.interpolate(
        imgL_,
        size=(imgL_.shape[2] // 2, imgL_.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.interpolate(
        imgR_,
        size=(imgL_.shape[2] // 2, imgL_.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgL_dw4 = F.interpolate(
        imgL_dw2,
        size=(imgL_dw2.shape[2] // 2, imgL_dw2.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw4 = F.interpolate(
        imgR_dw2,
        size=(imgR_dw2.shape[2] // 2, imgR_dw2.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=n_iters, flow_init=None)
    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iters // 2, flow_init=None)
    pred_flow = model(imgL_, imgR_, iters=n_iters // 4, flow_init=pred_flow_dw2)
    pred = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
    # pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    # print("Thops:")	
    # macs, params, ret_dict = thop.profile(model, inputs=(imgL_, imgR_), ret_layer_info=True)
    # print(ret_dict)
    # print(f"macs: {macs}")
    # print(f"params: {params}")

    # with open("../cre_stereo_model_layers.json", "w") as f:
    #    json.dump(ret_dict, f, indent=4)
    
    print("Torchinfo:")
    model_stats = summary(model, 
                          input_data=[imgL_dw4, imgR_dw4], 
                          col_names=("input_size", "output_size", "num_params", "mult_adds", "kernel_size"),
                          iters=n_iters, 
                          flow_init=None,
                          depth=summary_depth)
    summary_str = str(model_stats)
    with open(f"../cre_stereo_torchinfo_iter{n_iters}_depth{summary_depth}_dw4.txt", "w") as f:
        f.write(summary_str)

    model_stats = summary(model, 
                          input_data=[imgL_dw2, imgR_dw2], 
                          col_names=("input_size", "output_size", "num_params", "mult_adds", "kernel_size"),
                          iters=n_iters // 2, 
                          flow_init=pred_flow_dw4,
                          depth=summary_depth)
    summary_str = str(model_stats)
    with open(f"../cre_stereo_torchinfo_iter{n_iters // 2}_depth{summary_depth}_dw2.txt", "w") as f:
        f.write(summary_str)

    model_stats = summary(model, 
                          input_data=[imgL_, imgR_], 
                          col_names=("input_size", "output_size", "num_params", "mult_adds", "kernel_size"),
                          iters=n_iters // 4, 
                          flow_init=pred_flow_dw2,
                          depth=summary_depth)
    summary_str = str(model_stats)
    with open(f"../cre_stereo_torchinfo_iter{n_iters // 4}_depth{summary_depth}_full_scale.txt", "w") as f:
        f.write(summary_str)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-l','--left_image_fn', help='Filename of left image', required=True)
    parser.add_argument('-r','--right_image_fn', help='Filename of left image', required=True)
    parser.add_argument('-o', '--output_directory', help="Directory to save output", default="demo_output")
    parser.add_argument('-e', action='store_true', help="Use to compute params and MACs needed by network")
    args = parser.parse_args()
    
    left_img = cv2.imread(args.left_image_fn)
    right_img = cv2.imread(args.right_image_fn)


    in_h, in_w = left_img.shape[:2]

    if in_h%8 != 0:
        pad_h = in_h%8
        left_img  = np.pad(left_img,  ((pad_h//2, pad_h//2), (0, 0), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((pad_h//2, pad_h//2), (0, 0), (0, 0)), mode='reflect')
    
    if in_w%8 != 0:
        pad_w = in_w%8
        left_img  = np.pad(left_img,  ((0, 0), (pad_w//2, pad_w//2), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((0, 0), (pad_w//2, pad_w//2), (0, 0)), mode='reflect')        
    
    in_h, in_w = left_img.shape[:2]
    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h,in_w)
    assert eval_h%8 == 0, "input height should be divisible by 8"
    assert eval_w%8 == 0, "input width should be divisible by 8"
    
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

