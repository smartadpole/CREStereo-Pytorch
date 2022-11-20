import glob
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np
from dataset.dataset import CREStereoDataset
from typing import Optional
import cv2


IMG_EXTENSIONS= [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def kt_loader(filepath):

    left_path = os.path.join(filepath, 'image_2')
    right_path = os.path.join(filepath, 'image_3')
    displ_path = os.path.join(filepath, 'disp_occ_0')

    # total_name = sorted([name for name in os.listdir(left_path) if name.find('_10') > -1])
    total_name = [name for name in os.listdir(left_path) if name.find('_10') > -1]
    train_name = total_name[:160]
    val_name = total_name[160:]

    train_left = []
    train_right = []
    train_displ = []
    for name in train_name:
        train_left.append(os.path.join(left_path, name))
        train_right.append(os.path.join(right_path, name))
        train_displ.append(os.path.join(displ_path, name))

    val_left = []
    val_right = []
    val_displ = []
    for name in val_name:
        val_left.append(os.path.join(left_path, name))
        val_right.append(os.path.join(right_path, name))
        val_displ.append(os.path.join(displ_path, name))

    return train_left, train_right, train_displ, val_left, val_right, val_displ


def kt2012_loader(filepath):

    left_path = os.path.join(filepath, 'colored_0')
    right_path = os.path.join(filepath, 'colored_1')
    displ_path = os.path.join(filepath, 'disp_occ')

    total_name = sorted([name for name in os.listdir(left_path) if name.find('_10') > -1])
    train_name = total_name[:160]
    val_name = total_name[160:]

    train_left = []
    train_right = []
    train_displ = []
    for name in train_name:
        train_left.append(os.path.join(left_path, name))
        train_right.append(os.path.join(right_path, name))
        train_displ.append(os.path.join(displ_path, name))

    val_left = []
    val_right = []
    val_displ = []
    for name in val_name:
        val_left.append(os.path.join(left_path, name))
        val_right.append(os.path.join(right_path, name))
        val_displ.append(os.path.join(displ_path, name))

    return train_left, train_right, train_displ, val_left, val_right, val_displ


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class KITTIDataset(CREStereoDataset):

    def __init__(self, root, sub_indexes: Optional[np.ndarray] = None, eval_mode: bool = False):
        super().__init__(root, sub_indexes, eval_mode)
        self.imgs = glob.glob(os.path.join(root, "**/**/image_02/**/*.png"), recursive=False)
        if sub_indexes is not None and len(self.imgs) > 0:
            self.imgs = [self.imgs[idx] for idx in sub_indexes]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_item_paths(self, index):
        # find path
        left_path = self.imgs[index]
        right_path = left_path.replace('image_02', 'image_03')
        left_disp_path = os.path.splitext(left_path.replace('/image_02/', '/velodyne_points/'))[0] + '.bin'

        file_sources = {
            "left_path": left_path,
            "prefix": os.path.basename(left_path),
            "right_path": right_path,
            "left_disp_path": left_disp_path,
            "right_disp_path": ""
        }

        return file_sources

    def get_item(self, file_sources):
        # read img, disp
        left_img = cv2.imread(file_sources['left_path'], cv2.IMREAD_COLOR)
        right_img = cv2.imread(file_sources['right_path'], cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            return None, None, None, None

        left_disp = disparity_loader(file_sources['left_disp_path'])
        if left_disp is None:
            return None, None, None, None

        return left_img, right_img, left_disp, None