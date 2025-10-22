import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
from tqdm import tqdm

from Inference.base_options import BaseOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from VITON import viton_fullbody_seq as fullbody_seq
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model


def _infer_model_name(ckpt_dir: str, garment_name: str) -> str:
    """
    Reads checkpoints/<garment>/opt.txt and returns the recorded model name.
    Falls back to the legacy RNN name when no metadata is found.
    """
    opt_path = os.path.join(ckpt_dir, garment_name, "opt.txt")
    if not os.path.isfile(opt_path):
        return "pix2pixHD_RNN_RGBA"

    with open(opt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("model:"):
                return line.split(":", 1)[1].strip()
    return "pix2pixHD_RNN_RGBA"


def _build_pix2pix_loader(ckpt_dir: str, garment_name: str):
    """
    Returns a function compatible with viton_fullbody_seq.make_pix2pix_model,
    but loading checkpoints from `ckpt_dir` and respecting the current CUDA setup.
    """

    def _make_pix2pix_model(name, input_nc=6, output_nc=4, model_name=None):
        resolved_model_name = model_name or _infer_model_name(ckpt_dir, name)
        opt = TestOptions().parse(save=False, use_default=True, show_info=False)
        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.name = name
        opt.input_nc = input_nc
        opt.output_nc = output_nc
        opt.isTrain = False
        opt.model = resolved_model_name
        opt.checkpoints_dir = ckpt_dir
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
        model = create_model(opt)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    return _make_pix2pix_model


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA is unavailable on this system; full-body inference requires a GPU.")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but no GPU is detected.")
        return "cuda"
    raise RuntimeError("Full-body inference currently supports only CUDA execution.")


def _create_processor(garment_name: str, ckpt_dir: str):
    original_make_fn = fullbody_seq.make_pix2pix_model
    fullbody_seq.make_pix2pix_model = _build_pix2pix_loader(ckpt_dir, garment_name)
    try:
        processor = fullbody_seq.FullBodySeqFrameProcessor(garment_name)
    finally:
        fullbody_seq.make_pix2pix_model = original_make_fn
    return processor


def process_video(video_path: str, garment_name: str, ckpt_dir: str, output_path: str):
    video_loader = MultithreadVideoLoader(video_path, max_height=1024)
    video_writer = MultithreadVideoWriter(outvid=output_path, fps=video_loader.get_fps())
    frame_processor = _create_processor(garment_name, ckpt_dir)
    try:
        for _ in tqdm(range(len(video_loader))):
            frame = video_loader.cap()
            if frame is None:
                break
            result = frame_processor.forward(frame)
            video_writer.append(result)
        video_writer.make_video()
    finally:
        video_writer.close()
        video_loader.close()


if __name__ == "__main__":
    opts = BaseOptions()
    opt = opts.parse()
    _resolve_device(opt.device)
    if not os.path.isdir(opt.checkpoints_dir):
        raise FileNotFoundError(f"Checkpoint directory '{opt.checkpoints_dir}' does not exist.")
    process_video(
        video_path=opt.input_video,
        garment_name=opt.garment_name,
        ckpt_dir=opt.checkpoints_dir,
        output_path=opt.output_video,
    )
