import json
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

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


def gen_dataset(source_path, mask_dir, dataset_name, sequence_length=8, sequence_stride=1):
    assert sequence_length > 0, "sequence_length must be positive"
    assert sequence_stride > 0, "sequence_stride must be positive"

    video_loader = make_video_loader(source_path)
    smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=True)
    densepose_extractor = DensePoseExtractor()
    mask_lists = get_file_path_list(mask_dir, "png")
    assert len(mask_lists) == len(video_loader), "Number of masks and video frames are inconsistent!"

    full_body = FullBodySMPL()
    target_path = os.path.join("./PerGarmentDatasets", dataset_name)
    os.makedirs(target_path, exist_ok=True)

    dataset_height = None
    dataset_width = None
    saved_samples = []

    for frame_idx in tqdm(range(len(video_loader))):
        raw_image = video_loader.cap()
        raw_mask_path = mask_lists[frame_idx]
        if raw_image is None:
            break

        dataset_height = raw_image.shape[0]
        dataset_width = raw_image.shape[1]
        new_height = ROI_HEIGHT
        new_width = new_height * dataset_width // dataset_height
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
        raw_mask = cv2.resize(raw_mask, (dataset_width, dataset_height))

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

        sample_id = len(saved_samples)
        prefix = f"{sample_id:05d}"
        garment_path = os.path.join(target_path, f"{prefix}_garment.jpg")
        vm_path = os.path.join(target_path, f"{prefix}_vm.jpg")
        mask_path = os.path.join(target_path, f"{prefix}_mask.png")
        iuv_path = os.path.join(target_path, f"{prefix}_iuv.npy")
        trans2roi_path = os.path.join(target_path, f"{prefix}_trans2roi.npy")
        inv_path = os.path.join(target_path, f"{prefix}_inv_trans2roi.npy")

        cv2.imwrite(garment_path, roi_garment_img)
        cv2.imwrite(vm_path, raw_vm)
        cv2.imwrite(mask_path, raw_mask)
        np.save(iuv_path, raw_IUV)
        np.save(trans2roi_path, trans2roi)
        np.save(inv_path, inv_trans2roi)

        saved_samples.append(
            {
                "sample_id": prefix,
                "source_frame_index": frame_idx,
            }
        )

    dataset_info = {
        "height": dataset_height,
        "width": dataset_width,
        "roi_height": ROI_HEIGHT,
        "roi_width": ROI_WIDTH,
        "num_frames": len(saved_samples),
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
    }
    with open(os.path.join(target_path, "dataset_info.json"), "w") as outfile:
        json.dump(dataset_info, outfile)

    sequences = []
    if len(saved_samples) >= sequence_length:
        for start in range(0, len(saved_samples) - sequence_length + 1, sequence_stride):
            sequence_frames = saved_samples[start : start + sequence_length]
            sequences.append(
                {
                    "sequence_id": len(sequences),
                    "frames": [frame["sample_id"] for frame in sequence_frames],
                    "source_frame_indices": [frame["source_frame_index"] for frame in sequence_frames],
                }
            )
    sequence_manifest = {
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "num_sequences": len(sequences),
        "sequences": sequences,
    }
    with open(os.path.join(target_path, "sequences.json"), "w") as outfile:
        json.dump(sequence_manifest, outfile)
    video_loader.close()


def process_video(v_path, mask_dir, dataset_name, sequence_length=8, sequence_stride=1):
    gen_dataset(v_path, mask_dir, dataset_name, sequence_length=sequence_length, sequence_stride=sequence_stride)


if __name__ == "__main__":
    opts = BaseOptions()
    opts.initialize()
    opts.parser.add_argument("--sequence_length", type=int, default=8, help="number of frames per training sequence")
    opts.parser.add_argument("--sequence_stride", type=int, default=1, help="stride between consecutive sequences")
    opt = opts.parse()

    video_path = opt.video_path
    mask_dir = opt.mask_dir
    dataset_name = opt.dataset_name
    process_video(
        video_path,
        mask_dir,
        dataset_name=dataset_name,
        sequence_length=opt.sequence_length,
        sequence_stride=opt.sequence_stride,
    )
