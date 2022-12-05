import numpy as np
import torch
from torch.nn import functional as F

from torchinfo import summary


def inference(left, right, model, n_iter=20, init_flow: bool = True, device: str = 'cuda'):
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
    try:
        with torch.inference_mode():
            if init_flow:
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
                pred_flow = model(imgL, imgR, iters=n_iter // 2, flow_init=pred_flow_dw2)
            else:
                pred_flow = model(imgL, imgR, iters=n_iter, flow_init=None)
    except:
        with torch.no_grad():
            if init_flow:
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
                pred_flow = model(imgL, imgR, iters=n_iter // 2, flow_init=pred_flow_dw2)
            else:
                pred_flow = model(imgL, imgR, iters=n_iter, flow_init=None)

    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp


def evaluate_compute(model, imgL: np.ndarray, imgR: np.ndarray, device: str = 'cuda'):
    imgL_ = imgL.transpose(2, 0, 1)
    imgR_ = imgR.transpose(2, 0, 1)
    imgL_ = np.ascontiguousarray(imgL_[None, :, :, :])
    imgR_ = np.ascontiguousarray(imgR_[None, :, :, :])

    imgL_ = torch.tensor(imgL_.astype("float32")).to(device)
    imgR_ = torch.tensor(imgR_.astype("float32")).to(device)

    n_iters = 8
    summary_depth = 3
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
