import datetime
import os
import shutil
import sys
import time
import logging
from collections import namedtuple
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboardX import SummaryWriter

from nets import Model
from dataset.dataset import CREStereoDataset
from dataset.KITTI import KITTIDataset
from dataset.sceneflow_loader import SceneFlow, sf_loader_walk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dataset.KITTI as kt
from file import MkdirSimple

DATASET_TYPE = {
    'kitti': SceneFlow
    , 'crestereo': CREStereoDataset
    , 'sceneflow': SceneFlow
}


def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
    return args


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def adjust_learning_rate(optimizer, epoch):
    warm_up = 0.02
    const_range = 0.6
    min_lr_rate = 0.05

    if epoch <= args.n_total_epoch * warm_up:
        lr = (1 - min_lr_rate) * args.base_lr / (
                args.n_total_epoch * warm_up
        ) * epoch + min_lr_rate * args.base_lr
    elif args.n_total_epoch * warm_up < epoch <= args.n_total_epoch * const_range:
        lr = args.base_lr
    else:
        lr = (min_lr_rate - 1) * args.base_lr / (
                (1 - const_range) * args.n_total_epoch
        ) * epoch + (1 - min_lr_rate * const_range) / (1 - const_range) * args.base_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):
    '''
    valid: (2, 384, 512) (B, H, W) -> (B, 1, H, W)
    flow_preds[0]: (B, 2, H, W)
    flow_gt: (B, 2, H, W)
    '''
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        # todo : why not equal
        if flow_preds[i].shape != flow_gt.shape:
            continue
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = torch.abs(flow_preds[i] - flow_gt)
        flow_loss += i_weight * (valid.unsqueeze(1) * i_loss).mean()

    return flow_loss


def inference_eval(left, right, model, n_iter=20, init_flow=True, test_mode=False):
    # print("Model Forwarding...")
    imgL = left.type(torch.float32)
    imgR = right.type(torch.float32)
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
            pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2[-1])
        else:
            pred_flow = model(imgL, imgR, iters=n_iter, flow_init=None, test_mode=test_mode)

    return pred_flow


def test(model, imgL, imgR, disp_true):
    model.eval()
    imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        pred_disp = model(imgL, imgR, iters=20, flow_init=None, test_mode=True)

    final_disp = pred_disp.cpu()[:, 0, :, :]
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    print(index.shape)
    print(true_disp.shape)
    print(final_disp.shape)
    disp_true[index[0], index[1], index[2]] = np.abs(
        true_disp[index[0], index[1], index[2]] - final_disp[index[0], index[1], index[2]])
    correct = (disp_true[index[0], index[1], index[2]] < 3) | \
              (disp_true[index[0], index[1], index[2]] < true_disp[index[0], index[1], index[2]] * 0.05)

    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def eval_model(model, tb_log, worklog, epoch_idx, eval_iters, start_iters, total_iters, t0, dataloader_valid,
               minibatch_per_epoch, epoch_total_eval_loss):
    ##################
    ###  Evaluation  #
    ##################
    if epoch_idx % 50 == 0:
        t1_eval = time.perf_counter()
        for batch_idx, mini_batch_data in enumerate(dataloader_valid):
            if batch_idx % minibatch_per_epoch == 0 and batch_idx != 0:
                break
            if len(mini_batch_data["left"]) == 0:
                continue

            eval_iters += 1
            # loss = test(model, mini_batch_data["left"], mini_batch_data["right"], mini_batch_data["disparity"])

            # parse data
            left, right, gt_disp, valid_mask = (
                mini_batch_data["left"],
                mini_batch_data["right"],
                mini_batch_data["disparity"].cuda(),
                mini_batch_data["mask"].cuda(),
            )

            # pre-process
            gt_disp = torch.unsqueeze(gt_disp, dim=1)  # [2, h, w] -> [2, 1, h, w]
            gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)  # [2, 2, h, w]

            model.eval()
            pred_eval = inference_eval(left.cuda(), right.cuda(), model, n_iter=20, init_flow=False)
            t2_eval = time.perf_counter()

            loss_eval = sequence_loss(
                pred_eval, gt_flow, valid_mask, gamma=args.gamma
            )
            t3_eval = time.perf_counter()

            if batch_idx % (minibatch_per_epoch // 10) == 0:
                plt.close()
                pred_final = torch.squeeze(pred_eval[-1][0, 0, :, :])
                left_img = torch.squeeze(left[0, :, :, :]).permute(1, 2, 0)

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                im = axes[0, 0].imshow(np.squeeze(gt_disp[0, :, :, :].cpu().numpy()))
                axes[0, 0].set_title("disparity")

                divider = make_axes_locatable(axes[0, 0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                im = axes[0, 1].imshow(np.squeeze(pred_final.cpu().numpy()))
                axes[0, 1].set_title(f"pred disparity: {loss_eval.data.item():.02f}")
                divider = make_axes_locatable(axes[0, 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                axes[1, 0].imshow(np.squeeze(left_img.cpu().numpy()))
                axes[1, 0].set_title("left")

                pred_diff = np.squeeze(gt_disp[0, :, :, :].cpu().numpy()) - np.squeeze(pred_final.cpu().numpy())
                valid = np.squeeze(valid_mask[0, :, :].cpu().numpy()).astype(bool)
                pred_diff[~valid] = np.nan
                im = axes[1, 1].imshow(np.squeeze(pred_diff))
                axes[1, 1].set_title("error")

                divider = make_axes_locatable(axes[1, 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                plt.tight_layout()

                prefix = mini_batch_data["file_source"]["prefix"][0]

                tb_log.add_figure(f"Evaluation/{prefix}", fig,
                                  global_step=epoch_idx * len(dataloader_valid) + batch_idx)

            loss_item_eval = loss_eval.data.item()
            epoch_total_eval_loss += loss_item_eval

            if eval_iters % 10 == 0:
                tdata = t2_eval - t1_eval
                time_eval_passed = t3_eval - t0
                time_iter_passed = t3_eval - t1_eval
                step_passed = eval_iters - start_iters
                eta = (
                        (total_iters - eval_iters)
                        / max(step_passed, 1e-7)
                        * time_eval_passed
                )

                meta_info = list()
                meta_info.append("{:.2g} b/s".format(1.0 / time_eval_passed))
                meta_info.append("passed:{}".format(format_time(time_iter_passed)))
                meta_info.append("eta:{}".format(format_time(eta)))
                meta_info.append(
                    "data_time:{:.2g}".format(tdata / time_eval_passed)
                )

                meta_info.append(
                    "[{}/{}:{}/{}]".format(
                        epoch_idx,
                        args.n_total_epoch,
                        batch_idx,
                        minibatch_per_epoch,
                    )
                )
                loss_info = [" ==> {}:{:.4g}".format("eval loss", loss_item_eval)]
                # exp_name = ['\n' + os.path.basename(os.getcwd())]

                info = [",".join(meta_info)] + loss_info
                worklog.info("".join(info))

            t1_eval = time.perf_counter()

            tb_log.add_scalar(
                "valid/loss",
                epoch_total_eval_loss / minibatch_per_epoch,
                epoch_idx,
            )

    return eval_iters, epoch_total_eval_loss


def main(args):
    debug_image = False
    # initial info
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # rank, world_size = dist.get_rank(), dist.get_world_size()
    world_size = torch.cuda.device_count()  # number of GPU(s)

    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    # model / optimizer
    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    model = nn.DataParallel(model, device_ids=[i for i in range(world_size)])
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    # model = nn.DataParallel(model,device_ids=[0])

    tb_log = SummaryWriter(os.path.join(args.log_dir, "train.events"))

    # worklog
    logging.basicConfig(level=eval(args.log_level))
    worklog = logging.getLogger("train_logger")
    worklog.propagate = False
    fileHandler = logging.FileHandler(
        os.path.join(args.log_dir, "worklog.txt"), mode="a", encoding="utf8"
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    fileHandler.setFormatter(formatter)
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler.setFormatter(formatter)
    worklog.handlers = [fileHandler, consoleHandler]

    # params stat
    worklog.info(f"Use {world_size} GPU(s)")
    worklog.info("Params: %s" % sum([p.numel() for p in model.parameters()]))

    # load pretrained model if exist
    chk_path = os.path.join(log_model_dir, "latest.pth")
    if args.loadmodel is not None:
        chk_path = args.loadmodel
    elif not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        # if rank == 0:
        worklog.info(f"loading model: {chk_path}")
        state_dict = torch.load(chk_path)
        if 'state_dict' in state_dict.keys():
            model.load_state_dict(state_dict['state_dict'])
            optimizer.load_state_dict(state_dict['optim_state_dict'])
            resume_epoch_idx = state_dict["epoch"]
            resume_iters = state_dict["iters"]
            start_epoch_idx = resume_epoch_idx + 1
            start_iters = resume_iters
        else:
            start_epoch_idx = 1
            start_iters = 0
            model_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # add `module.`
                model_state_dict[name] = v
            model.load_state_dict(model_state_dict)
    else:
        start_epoch_idx = 1
        start_iters = 0

    # datasets
    datasets_train = []
    dataset_eval = None

    dataset_type_list = args.dataset
    training_data_path_list = args.training_data_path
    minibatch_per_epoch_list = args.minibatch_per_epoch
    if not isinstance(args.dataset, list):
        dataset_type_list = [args.dataset, ]
    if not isinstance(training_data_path_list, list):
        training_data_path_list = [training_data_path_list, ]
    if not isinstance(minibatch_per_epoch_list, list):
        minibatch_per_epoch_list = [minibatch_per_epoch_list, ]

    for i, type in enumerate(dataset_type_list):
        if type not in DATASET_TYPE.keys():
            print("donot support dataset type: {}".format(dataset_type_list))
            return

        if 'kitti' == type:
            all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt.kt_loader(training_data_path_list[i])
            datasets_train.append(
                DATASET_TYPE[type](all_limg, all_rimg, all_ldisp, training=True, dploader=kt.disparity_loader_real))
            dataset_eval = DATASET_TYPE[type](test_limg, test_rimg, test_ldisp, training=False,
                                              dploader=kt.disparity_loader_real)
        elif 'sceneflow' == type:
            all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = sf_loader_walk(training_data_path_list[i])
            datasets_train.append(DATASET_TYPE[type](all_limg, all_rimg, all_ldisp, training=True))
            # dataset_eval = DATASET_TYPE[type](test_limg, test_rimg, test_ldisp, training=False)
        else:
            if 'sceneflow' in dataset_type_list or 'kitti' in dataset_type_list:
                datasets_train.append(DATASET_TYPE[type](training_data_path_list[i]))
            else:
                dataset = DATASET_TYPE[type](training_data_path_list[i])

                # Creating data indices for training and validation splits:
                dataset_size = len(dataset)
                indices = list(range(dataset_size))
                validation_split = .05
                split = int(np.floor(validation_split * dataset_size))

                np.random.seed(1234)
                np.random.shuffle(indices)
                train_indices, val_indices = indices[split:], indices[:split]

                datasets_train.append(DATASET_TYPE[type](training_data_path_list[i], sub_indexes=train_indices))
                dataset_eval = DATASET_TYPE[type](training_data_path_list[i], sub_indexes=val_indices,
                                                  eval_mode=True)  # No augmentation

    # Creating PT data samplers and loaders:

    # if rank == 0:
    worklog.info("Dataset size: {} = {}".format(sum([len(d) for d in datasets_train]),
                                                ' + '.join([str(len(d)) for d in datasets_train])))
    dataloader_train = [DataLoader(d, args.batch_size, shuffle=True,
                                   num_workers=args.num_works, drop_last=True, persistent_workers=False,
                                   pin_memory=True) for d in datasets_train]
    dataloader_valid = DataLoader(dataset_eval, args.batch_size, shuffle=True,
                                  num_workers=args.num_works, drop_last=True, persistent_workers=False,
                                  pin_memory=True)

    # counter
    cur_iters = start_iters
    eval_iters = start_iters
    minibatch_per_epoch = sum(minibatch_per_epoch_list)
    total_iters = minibatch_per_epoch * args.n_total_epoch
    t0 = time.perf_counter()
    for epoch_idx in range(start_epoch_idx, args.n_total_epoch + 1):
        torch.manual_seed(args.seed + epoch_idx)
        torch.cuda.manual_seed(args.seed + epoch_idx)

        # adjust learning rate
        epoch_total_train_loss = 0
        adjust_learning_rate(optimizer, epoch_idx)
        model.train()

        t1 = time.perf_counter()

        epoch_total_eval_loss = 0.
        # eval_iters, epoch_total_eval_loss = eval_model(model, tb_log, worklog, epoch_idx, eval_iters, start_iters,
        #                                                total_iters, t0, dataloader_valid,
        #                                                minibatch_per_epoch, epoch_total_eval_loss)

        # for mini_batch_data in dataloader:
        batch_idx = -1
        for id_data, dl in enumerate(dataloader_train):
            for sub_batch_idx, mini_batch_data in enumerate(dl):
                batch_idx += 1
                if sub_batch_idx % minibatch_per_epoch_list[id_data] == 0 and sub_batch_idx != 0:
                    break
                if len(mini_batch_data["left"]) == 0:
                    continue

                cur_iters += 1

                # parse data
                left, right, gt_disp, valid_mask = (
                    mini_batch_data["left"],
                    mini_batch_data["right"],
                    mini_batch_data["disparity"].cuda(),
                    mini_batch_data["mask"].cuda(),
                )

                t2 = time.perf_counter()
                optimizer.zero_grad()

                # pre-process
                gt_disp = torch.unsqueeze(gt_disp, dim=1)  # [2, h, w] -> [2, 1, h, w]
                gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)  # [2, 2, h, w]

                # forward
                flow_predictions = model(left.cuda(), right.cuda())

                # loss & backword
                loss = sequence_loss(
                    flow_predictions, gt_flow, valid_mask, gamma=args.gamma
                )

                # loss stats
                loss_item = loss.data.item()
                epoch_total_train_loss += loss_item
                loss.backward()
                optimizer.step()
                t3 = time.perf_counter()

                if cur_iters % 10 == 0:
                    tdata = t2 - t1
                    time_train_passed = t3 - t0
                    time_iter_passed = t3 - t1
                    step_passed = cur_iters - start_iters
                    eta = ((total_iters - cur_iters) / max(step_passed, 1e-7) * time_train_passed)

                    meta_info = list()
                    meta_info.append("{:.2g} b/s".format(1.0 / time_iter_passed))
                    meta_info.append("passed:{}".format(format_time(time_train_passed)))
                    meta_info.append("eta:{}".format(format_time(eta)))
                    meta_info.append("data_time:{:.2g}".format(tdata / time_iter_passed))

                    meta_info.append("lr:{:.5g}".format(optimizer.param_groups[0]["lr"]))
                    meta_info.append(
                        "[{}/{}:{}/{}]".format(epoch_idx, args.n_total_epoch, batch_idx, minibatch_per_epoch, ))
                    loss_info = [" ==> {}:{:.4g}".format("loss", loss_item)]
                    # exp_name = ['\n' + os.path.basename(os.getcwd())]

                    info = [",".join(meta_info)] + loss_info
                    worklog.info("".join(info))

                    # minibatch loss
                    tb_log.add_scalar("train/loss_batch", loss_item, cur_iters)
                    tb_log.add_scalar("train/lr", optimizer.param_groups[0]["lr"], cur_iters)
                    tb_log.flush()

                t1 = time.perf_counter()

        tb_log.add_scalar(
            "train/loss",
            epoch_total_train_loss / minibatch_per_epoch,
            epoch_idx,
        )

        # save model params
        ckp_data = {
            "epoch": epoch_idx,
            "iters": cur_iters,
            "eval_iters": eval_iters,
            "batch_size": args.batch_size,
            "epoch_size": minibatch_per_epoch,
            "train_loss": epoch_total_train_loss / minibatch_per_epoch,
            "eval_loss": epoch_total_eval_loss / minibatch_per_epoch,
            "state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }

        torch.save(ckp_data, os.path.join(log_model_dir, "latest.pth"))
        if epoch_idx % args.model_save_freq_epoch == 0:
            save_path = os.path.join(log_model_dir, "epoch-%d.pth" % epoch_idx)
            worklog.info(f"Model params saved: {save_path}")
            torch.save(ckp_data, save_path)

        eval_iters, epoch_total_eval_loss = eval_model(model, tb_log, worklog, epoch_idx, eval_iters, start_iters,
                                                       total_iters, t0, dataloader_valid,
                                                       minibatch_per_epoch, epoch_total_eval_loss)

        tb_log.flush()

        worklog.info(f"Epoch is done, next epoch.")

    worklog.info("Training is done, exit.")


if __name__ == "__main__":
    # train configuration
    config_file = "cfgs/train.yaml"
    args = parse_yaml(config_file)
    MkdirSimple(args.log_dir)
    date = datetime.date.today().strftime("20%y-%m-%d_") + format_time(time.perf_counter())
    shutil.copyfile(config_file, os.path.join(args.log_dir, 'train_{}.yaml'.format(date)))

    main(args)
