import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from generate_depth_indemind import gray_scale_region
from tqdm.contrib import tzip
from file import MkdirSimple

def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--file_name', help='image list in file', required=True)
    parser.add_argument('--depth_dir', type=str, default='DEPTH/')
    parser.add_argument('--dest_dir', type=str, default='DEPTH/CREStereo_delete')
    args = parser.parse_args()

    return args

def get_min_max_index(image, max_index_ratio):

    hist = cv2.calcHist([image],[0],None,[256],[0,255])
    hist = hist.ravel()
    min_index = 0
    max_index = 255

    for i in range(len(hist)):
        if hist[i] > 0:
            min_index = max(0, i - 1)
            break
    for i in range(min_index, len(hist)):
        if hist[i] < 100:
            max_index = min(i + 1, 255)
            max_ration = np.sum(hist[max_index:])/(image.shape[0]*image.shape[1])
            if max_ration > max_index_ratio:
                continue
            else:
                break
    return min_index, max_index



def get_file_name(file_name, image_path, depth_dir, dest_dir):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    image_list = []
    depth_list = []
    dest_list = []
    for image_name in file_lists:
        image_current = image_name.split()[1]
        image_full_path = os.path.join(image_path, image_current)
        image_dest_path = image_full_path.replace("REMAP", depth_dir).split()[0]
        image_dest_path = image_dest_path.replace(".jpg", ".png")
        image_scale_dest_path = image_dest_path.replace(depth_dir, dest_dir)
        image_list.append(image_full_path)
        depth_list.append(image_dest_path)
        dest_list.append(image_scale_dest_path)

    return image_list, depth_list, dest_list, file_lists

def filter_too_close_origin_depth(image, value=240):
    index = image > value
    ratio = (np.sum(index)) / (image.shape[0] * image.shape[1])
    return ratio

def get_max_min_value(image, image_points):
    min_value = np.min(image)
    max_value = min_value
    hist = cv2.calcHist([image],[0],None,[256],[0,255])
    pd.DataFrame(hist)
    return min_value, max_value
if __name__ == '__main__':
    args = GetArgs()

    image_list, depth_list, dest_list, file_lists = get_file_name(args.file_name, args.data_path, args.depth_dir, args.dest_dir)

    filter_value = 150
    filter_ratio = 0.1
    max_index_ratio = 0.1

    print("filter_value: ", filter_value)
    print("filter_ratio: ", filter_ratio)
    print("max_index_ratio: ", max_index_ratio)
    all_image = len(image_list)
    save_image = 0
    save_image_id = []
    for i, (image_name, scale_image_name) in enumerate(tzip(depth_list, dest_list)):
        image = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)

        if filter_too_close_origin_depth(image, filter_value) > filter_ratio:
            print("\nimage too close > ", filter_value, " ratio > ", filter_ratio, image_name)
            continue
        else:
            min_index, max_index = get_min_max_index(image, max_index_ratio)
            if max_index - min_index < 2:
                print("max_index - min_index < ", 2, ", continue")
            scale_image = gray_scale_region(image, min_index, max_index)
            MkdirSimple(scale_image_name)
            cv2.imwrite(scale_image_name, scale_image)
            save_image += 1
            save_image_id.append(i)
    with open("file_list.txt", "w") as f:
        for file_list in file_lists:
            f.write(file_list)
    print("all_image: ", all_image, "\nsave_image: ", save_image, "file_list write: ", len(file_list))
