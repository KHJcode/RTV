import json
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from util.cv2_trans_util import get_inverse_trans
from util.densepose_util import IUV2SSDP


class FullBodyGarment(data.Dataset):
    def __init__(self, path: str, img_size: int = 576):
        self.img_dir = path
        self.image_list = self._get_image_list()
        with open(os.path.join(self.img_dir, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)
        self.raw_height = dataset_info["height"]
        self.raw_width = dataset_info["width"]
        self.roi_height = dataset_info.get("roi_height", 1024)
        self.roi_width = dataset_info.get("roi_width", 768)
        self.aspect_ratio = self.roi_width / self.roi_height

        target_height = img_size
        target_width = max(1, int(round(target_height * self.aspect_ratio)))
        self.transform = transforms.Compose(
            [
                transforms.Resize((target_height, target_width)),
                transforms.ToTensor(),
            ]
        )
        self.randomaffine = RandomAffineMatrix(
            degrees=20,
            translate=(0.2, 0.2),
            scale=(0.8, 1.4),
            shear=(-5, 5, -5, 5),
            roi_size=(self.roi_width, self.roi_height),
        )

    def __getitem__(self, index):
        garment_path = self.image_list[index]
        garment_img = np.array(Image.open(garment_path))
        raw_h, raw_w = self.raw_height, self.raw_width

        base_name = os.path.basename(garment_path).split("_")[0]
        trans2roi_path = os.path.join(self.img_dir, f"{base_name}_trans2roi.npy")
        trans2roi = np.load(trans2roi_path)
        inv_trans_path = os.path.join(self.img_dir, f"{base_name}_inv_trans2roi.npy")
        if os.path.exists(inv_trans_path):
            inv_trans = np.load(inv_trans_path)
        else:
            inv_trans = get_inverse_trans(trans2roi)

        garment_img = cv2.warpAffine(
            garment_img,
            inv_trans,
            (raw_w, raw_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        vm_path = os.path.join(self.img_dir, f"{base_name}_vm.jpg")
        vm_img = np.array(Image.open(vm_path))
        vm_img = cv2.resize(vm_img, (raw_w, raw_h))

        mask_path = os.path.join(self.img_dir, f"{base_name}_mask.png")
        mask_img = np.array(Image.open(mask_path))
        mask_img = cv2.resize(mask_img, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

        iuv_path = os.path.join(self.img_dir, f"{base_name}_iuv.npy")
        IUV = np.load(iuv_path)
        dp_img = IUV2SSDP(IUV)
        dp_img = cv2.resize(dp_img, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

        new_trans2roi = self.randomaffine(trans2roi)

        roi_garment_img = cv2.warpAffine(
            garment_img,
            new_trans2roi,
            (self.roi_width, self.roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        roi_dp_img = cv2.warpAffine(
            dp_img,
            new_trans2roi,
            (self.roi_width, self.roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        roi_vm_img = cv2.warpAffine(
            vm_img,
            new_trans2roi,
            (self.roi_width, self.roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        roi_mask_img = cv2.warpAffine(
            mask_img,
            new_trans2roi,
            (self.roi_width, self.roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )

        if self.transform is not None:
            roi_garment_img = self.transform(Image.fromarray(roi_garment_img))
            roi_vm_img = self.transform(Image.fromarray(roi_vm_img))
            roi_mask_img = self.transform(Image.fromarray(roi_mask_img))
            roi_dp_img = self.transform(Image.fromarray(roi_dp_img))

        if torch.cuda.is_available():
            roi_garment_img = roi_garment_img.cuda()
            roi_vm_img = roi_vm_img.cuda()
            roi_mask_img = roi_mask_img.cuda()
            roi_dp_img = roi_dp_img.cuda()

        return (
            self._normalize(roi_garment_img),
            self._normalize(roi_vm_img),
            self._normalize(roi_dp_img),
            roi_mask_img,
        )

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2.0 - 1.0

    def _get_image_list(self):
        image_list = []
        for item in sorted(os.listdir(self.img_dir)):
            if item.endswith("_garment.jpg"):
                image_list.append(os.path.join(self.img_dir, item))
        return image_list

    def __len__(self):
        return len(self.image_list)


class RandomAffineMatrix:
    def __init__(
        self,
        degrees: float,
        translate: Tuple[float, float] = None,
        scale: Tuple[float, float] = None,
        shear: Tuple[float, float, float, float] = None,
        roi_size: Tuple[int, int] = (768, 1024),
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.roi_width, self.roi_height = roi_size
        self.center = np.array([self.roi_width / 2.0, self.roi_height / 2.0])

    def __call__(self, trans: np.ndarray):
        R_hat, t_hat = get_random_affine_params(
            self.degrees, self.translate, self.scale, self.shear, self.roi_width, self.roi_height
        )
        return self.deform(R_hat, t_hat, trans)

    def batch_forward(self, trans_list):
        R_hat, t_hat = get_random_affine_params(
            self.degrees, self.translate, self.scale, self.shear, self.roi_width, self.roi_height
        )
        return [self.deform(R_hat, t_hat, trans) for trans in trans_list]

    def deform(self, R_hat: np.ndarray, t_hat: np.ndarray, trans: np.ndarray):
        R = trans[:, :2]
        t = trans[:, 2]
        R_new = np.dot(R_hat, R)
        t_new = np.dot(R_hat, t - self.center) + t_hat + self.center
        new_trans = np.concatenate((R_new, t_new[:, None]), axis=1)
        return new_trans


def get_random_affine_params(
    degrees,
    translate,
    scale,
    shear,
    roi_width,
    roi_height,
):
    angle = np.random.uniform(-degrees, degrees)
    angle_rad = np.deg2rad(angle)

    if translate is not None:
        max_dx = translate[0] * roi_width
        max_dy = translate[1] * roi_height
        tx = np.random.uniform(-max_dx, max_dx)
        ty = np.random.uniform(-max_dy, max_dy)
    else:
        tx, ty = 0.0, 0.0

    if scale is not None:
        scale_factor = np.random.uniform(scale[0], scale[1])
    else:
        scale_factor = 1.0

    if shear is not None:
        shear_x = np.random.uniform(shear[0], shear[1])
        shear_y = np.random.uniform(shear[2], shear[3]) if len(shear) > 2 else 0.0
    else:
        shear_x, shear_y = 0.0, 0.0

    cos_theta = np.cos(angle_rad) * scale_factor
    sin_theta = np.sin(angle_rad) * scale_factor

    shear_x_rad = np.deg2rad(shear_x)
    shear_y_rad = np.deg2rad(shear_y)

    M = np.array(
        [
            [cos_theta + np.tan(shear_y_rad) * sin_theta, -sin_theta + np.tan(shear_y_rad) * cos_theta],
            [sin_theta + np.tan(shear_x_rad) * cos_theta, cos_theta + np.tan(shear_x_rad) * sin_theta],
        ]
    )

    t = np.array([tx, ty])

    return M, t
