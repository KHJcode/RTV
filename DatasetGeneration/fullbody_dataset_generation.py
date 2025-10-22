import os
import sys
import json
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from util.torch_device import ensure_compatible_cuda

ensure_compatible_cuda()

from tqdm import tqdm

from DatasetGeneration.options import BaseOptions
from SMPL.fullbody_smpl.FullBody import FullBodySMPL
from SMPL.smpl_regressor import SMPL_Regressor
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.file_io import get_file_path_list


ROI_HEIGHT = 1024
ROI_WIDTH = 768
ROI_SCALE = 1.4


def make_video_loader(source_path):
    from util.multithread_video_loader import MultithreadVideoLoader

    source_dataset_new = MultithreadVideoLoader(source_path, max_height=3072)
    print(f"Length of source dataset: {len(source_dataset_new)}")
    return source_dataset_new


def gen_dataset(source_path, mask_dir, dataset_name):
    video_loader = make_video_loader(source_path)
    smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=True)
    densepose_extractor = DensePoseExtractor()
    mask_lists = get_file_path_list(mask_dir, "png")
    assert len(mask_lists) == len(video_loader), "Number of masks and video frames are inconsistent!"

    full_body = FullBodySMPL()
    target_path = os.path.join("./PerGarmentDatasets", dataset_name)
    os.makedirs(target_path, exist_ok=True)

    height = None
    width = None

    for i in tqdm(range(len(video_loader))):
        raw_image = video_loader.cap()
        raw_mask_path = mask_lists[i]
        if raw_image is None:
            break

        height = raw_image.shape[0]
        width = raw_image.shape[1]
        new_height = ROI_HEIGHT
        new_width = new_height * width // height
        resized_image = cv2.resize(raw_image, (new_width, new_height))

        smpl_param = smpl_regressor.forward(raw_image, roi=False)
        if smpl_param is None:
            continue
        trans2roi, inv_trans2roi = smpl_regressor.get_fullbody_trans2roi(
            smpl_param, s=ROI_SCALE, new_h=ROI_HEIGHT, new_w=ROI_WIDTH
        )

        vertices = smpl_regressor.get_raw_verts(smpl_param)

        raw_vm = full_body.render(vertices, height=new_height, width=new_width)

        raw_IUV = densepose_extractor.get_IUV(resized_image, isRGB=False)
        if raw_IUV is None:
            continue

        raw_mask = cv2.imread(raw_mask_path, cv2.IMREAD_UNCHANGED)[:, :, 3]
        raw_mask = cv2.resize(raw_mask, (width, height))

        raw_garment = raw_image.copy()
        raw_garment[raw_mask < 127] = 0
        roi_garment_img = cv2.warpAffine(
            raw_garment,
            trans2roi,
            (ROI_WIDTH, ROI_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        garment_path = os.path.join(target_path, f"{i:05d}_garment.jpg")
        vm_path = os.path.join(target_path, f"{i:05d}_vm.jpg")
        mask_path = os.path.join(target_path, f"{i:05d}_mask.png")
        iuv_path = os.path.join(target_path, f"{i:05d}_iuv.npy")
        trans2roi_path = os.path.join(target_path, f"{i:05d}_trans2roi.npy")
        inv_path = os.path.join(target_path, f"{i:05d}_inv_trans2roi.npy")

        cv2.imwrite(garment_path, roi_garment_img)
        cv2.imwrite(vm_path, raw_vm)
        cv2.imwrite(mask_path, raw_mask)
        np.save(iuv_path, raw_IUV)
        np.save(trans2roi_path, trans2roi)
        np.save(inv_path, inv_trans2roi)

    dataset_info = {
        "height": height,
        "width": width,
        "roi_height": ROI_HEIGHT,
        "roi_width": ROI_WIDTH,
    }
    with open(os.path.join(target_path, "dataset_info.json"), "w") as outfile:
        json.dump(dataset_info, outfile)
    video_loader.close()


def process_video(v_path, mask_dir, dataset_name):
    gen_dataset(v_path, mask_dir, dataset_name)


if __name__ == "__main__":
    opts = BaseOptions()
    opt = opts.parse()
    video_path = opt.video_path
    mask_dir = opt.mask_dir
    dataset_name = opt.dataset_name
    process_video(video_path, mask_dir, dataset_name=dataset_name)
