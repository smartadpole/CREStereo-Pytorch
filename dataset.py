import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageEnhance
from typing import Optional

from megengine.data.dataset import Dataset


class Augmentor:
    def __init__(
        self,
        image_height=384,
        image_width=512,
        max_disp=256,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def __call__(self, left_img, right_img, left_disp):
        # 1. chromatic augmentation
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 2.2) random resize
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        disp_mask = disp_mask.astype("float32")
        disp_mask = cv2.resize(
            disp_mask,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        left_disp = (
            cv2.resize(
                left_disp,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )

        # 2.3) random crop
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(
            left_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        right_img = cv2.warpAffine(
            right_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        left_disp = cv2.warpAffine(
            left_disp,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        disp_mask = cv2.warpAffine(
            disp_mask,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        return left_img, right_img, left_disp, disp_mask


class CREStereoDataset(Dataset):
    def __init__(self, root, sub_indexes: Optional[np.ndarray] = None, eval_mode: bool = False):
        super().__init__()
        self.imgs = glob.glob(os.path.join(root, "**/*_left.jpg"), recursive=True)
        if sub_indexes is not None:
            self.imgs = [self.imgs[idx] for idx in sub_indexes]

        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=256,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
        self.eval_mode = eval_mode

    def get_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            return None
        return disp.astype(np.float32) / 32

    def __getitem__(self, index):
        # find path
        left_path = self.imgs[index]
        prefix = left_path[: left_path.rfind("_")]
        right_path = prefix + "_right.jpg"
        left_disp_path = prefix + "_left.disp.png"
        right_disp_path = prefix + "_right.disp.png"

        # read img, disp
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            return {
                "left": [],
                "right": [],
                "disparity": [],
                "mask": [],
            }

        left_disp = self.get_disp(left_disp_path)
        right_disp = self.get_disp(right_disp_path)
        if left_disp is None or right_disp is None:
            return {
                "left": [],
                "right": [],
                "disparity": [],
                "mask": [],
            }

        if not self.eval_mode:
            if self.rng.binomial(1, 0.5):
                left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
                left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
            left_disp[left_disp == np.inf] = 0

            # augmentaion
            left_img, right_img, left_disp, disp_mask = self.augmentor(
                left_img, right_img, left_disp
            )
        else:
            resize_scale = 1.
            disp_mask = (left_disp < float(self.augmentor.max_disp / resize_scale)) & (left_disp > 0)
            disp_mask = disp_mask.astype("float32")

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        file_sources = {
            "left_path": left_path,
            "prefix": os.path.basename(prefix),
            "right_path": right_path,
            "left_disp_path": left_disp_path,
            "right_disp_path": right_disp_path
        }
        return {
            "file_source": file_sources,
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.imgs)
