import os
from PIL import Image
from dataset import readpfm as rp
from file import Walk
import cv2
from dataset.dataset import CREStereoDataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def sf_loader_walk(filepath):
    if os.path.exists(os.path.join(filepath, 'all.txt')):
        all_file = [l.strip('\n').strip() for l in open(os.path.join(filepath, 'all.txt')).readlines()]
    else:
        all_file = Walk(filepath, ['jpg', 'jpeg', 'png', 'bmp', 'pfm'])

    all_left_img = [f for f in all_file if "/left/" in f and "/frames_cleanpass/" in f]

    left_img = []
    right_img = []
    left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    for f in all_left_img:
        r = f.replace('left/', 'right/')
        d = os.path.splitext(f.replace('frames_cleanpass/', 'disparity/'))[0] + '.pfm'
        if r in all_file and d in all_file:
            if '/TEST/' in f:
                test_left_img.append(f)
                test_right_img.append(r)
                test_left_disp.append(d)
            else:
                left_img.append(f)
                right_img.append(r)
                left_disp.append(d)

    return left_img, right_img, left_disp, test_left_img, test_right_img, test_left_disp


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class SceneFlow(CREStereoDataset):

    def __init__(self, left, right, left_disp, training, imgloader=img_loader, dploader=disparity_loader):
        super().__init__("", None, not training)
        self.left = left
        self.right = right
        self.disp_L = left_disp
        self.imgloader = imgloader
        self.dploader = dploader
        self.training = training

    def get_item_paths(self, index):
        # find path
        left_path = self.left[index]
        right_path = self.right[index]
        left_disp_path = self.disp_L[index]

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

        left_disp, scale = self.dploader(file_sources['left_disp_path'])
        if left_disp is None:
            return None, None, None, None

        return left_img, right_img, left_disp.copy(), None

    def __len__(self):
        return len(self.left)
