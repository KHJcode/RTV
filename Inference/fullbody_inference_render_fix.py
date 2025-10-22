import os
import sys
from contextlib import contextmanager
from typing import Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from Inference.base_options import BaseOptions
from VITON import viton_fullbody_seq as fullbody_seq
from model.pix2pixHD.models import create_model
from options.test_options import TestOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter


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


def _refine_soft_alpha(raw_alpha: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """
    Produces a soft alpha mask that keeps confident regions opaque while
    smoothing thin boundaries and clamping the contour to the original mask.
    """
    mask_u8 = raw_alpha.astype(np.uint8)
    if mask_u8.size == 0:
        return np.zeros_like(mask_u8, dtype=np.float32)

    alpha = mask_u8.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.6, sigmaY=0.6)

    binary = base_mask.astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)

    alpha = np.clip(alpha, 0.0, 1.0)
    alpha *= binary.astype(np.float32)
    alpha = np.where(alpha < 0.01, 0.0, alpha)
    alpha = np.where(alpha > 0.98, 1.0, alpha)
    return alpha


def _suppress_boundary_artifacts(
    foreground: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
    base_mask: np.ndarray,
    dark_threshold: float = 42.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes residual dark speckles that typically appear along the garment outline
    when RGB pixels with low intensity are paired with uncertain alpha.
    """
    refined_alpha = np.clip(alpha, 0.0, 1.0)
    if refined_alpha.size == 0:
        return foreground, refined_alpha

    refined_alpha = cv2.GaussianBlur(refined_alpha.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    refined_alpha = np.clip(refined_alpha, 0.0, 1.0)

    binary = (base_mask > 0).astype(np.uint8)
    if not np.any(binary):
        return foreground, refined_alpha

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel3, iterations=1)
    dilated = cv2.dilate(binary, kernel3, iterations=1)
    boundary = (dilated > 0) & (eroded == 0)

    # Compute luminance to detect dark halos; consider very low per-channel values too.
    luminance = (
        foreground[:, :, 0].astype(np.float32) * 0.299
        + foreground[:, :, 1].astype(np.float32) * 0.587
        + foreground[:, :, 2].astype(np.float32) * 0.114
    )
    min_channel = foreground.min(axis=2)

    uncertain_zone = (refined_alpha > 0.05) & (refined_alpha < 0.5)
    artifact_mask = boundary & uncertain_zone & (
        (luminance < dark_threshold) | (min_channel < dark_threshold * 0.4)
    )
    if not np.any(artifact_mask):
        return foreground, refined_alpha

    cleaned = foreground.copy()
    bg_blur = cv2.GaussianBlur(background, (0, 0), sigmaX=1.2, sigmaY=1.2)
    cleaned[artifact_mask] = bg_blur[artifact_mask]

    refined_alpha = refined_alpha.copy()
    refined_alpha[artifact_mask] = 0.0
    refined_alpha = cv2.GaussianBlur(refined_alpha.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    refined_alpha = np.clip(refined_alpha, 0.0, 1.0)
    refined_alpha *= base_mask.astype(np.float32)
    return cleaned, refined_alpha


def _artifact_aware_overlay(raw_img: np.ndarray, raw_target: np.ndarray, raw_alpha: np.ndarray) -> np.ndarray:
    """
    Drop-in replacement for naive_overlay_alpha that keeps the rendering logic intact
    while cleaning up boundary artifacts before composing the frame.
    """
    base_mask = (raw_alpha.astype(np.uint8) > 160).astype(np.uint8)
    if base_mask.sum() == 0:
        alpha = np.clip(raw_alpha.astype(np.float32) / 255.0, 0.0, 1.0)
        bg = raw_img.astype(np.float32)
        fg = raw_target.astype(np.float32)
        composed = fg * alpha[..., None] + bg * (1.0 - alpha[..., None])
        return np.clip(composed, 0.0, 255.0).astype(np.uint8)
    soft_alpha = _refine_soft_alpha(raw_alpha, base_mask)
    cleaned_target, refined_alpha = _suppress_boundary_artifacts(raw_target, soft_alpha, raw_img, base_mask)

    bg = raw_img.astype(np.float32)
    fg = cleaned_target.astype(np.float32)
    alpha_expanded = refined_alpha[..., None]
    premultiplied_fg = fg * alpha_expanded

    garment_mask = (alpha_expanded > 1e-3).astype(np.uint8)
    if garment_mask.any():
        smooth_fg = cv2.bilateralFilter(premultiplied_fg.astype(np.float32), d=5, sigmaColor=10.0, sigmaSpace=2.5)
        premultiplied_fg = np.where(garment_mask == 1, smooth_fg, premultiplied_fg)

    composed = premultiplied_fg + bg * (1.0 - alpha_expanded)

    # Reinstate fine garment details where alpha is confident to avoid over-smoothing.
    high_conf_mask = (refined_alpha > 0.75).astype(np.float32)[..., None]
    if np.any(high_conf_mask):
        original_premult = fg * (soft_alpha[..., None])
        composed = composed * (1.0 - high_conf_mask) + (original_premult + bg * (1.0 - soft_alpha[..., None])) * high_conf_mask
    edge_mask = cv2.Canny((refined_alpha * 255.0).astype(np.uint8), 20, 60) > 0
    if np.any(edge_mask):
        composed_blur = cv2.GaussianBlur(composed, (0, 0), sigmaX=1.1, sigmaY=1.1)
        composed[edge_mask] = composed_blur[edge_mask]

    return np.clip(composed, 0.0, 255.0).astype(np.uint8)


@contextmanager
def _patched_overlay():
    from composition import naive_overlay as naive_module
    from VITON import viton_fullbody_seq as fullbody_seq_module

    original = naive_module.naive_overlay_alpha
    original_seq = getattr(fullbody_seq_module, "naive_overlay_alpha", original)
    naive_module.naive_overlay_alpha = _artifact_aware_overlay
    fullbody_seq_module.naive_overlay_alpha = _artifact_aware_overlay
    try:
        yield
    finally:
        naive_module.naive_overlay_alpha = original
        fullbody_seq_module.naive_overlay_alpha = original_seq


def _create_processor(garment_name: str, ckpt_dir: str):
    original_make_fn = fullbody_seq.make_pix2pix_model
    fullbody_seq.make_pix2pix_model = _build_pix2pix_loader(ckpt_dir, garment_name)
    try:
        processor = fullbody_seq.FullBodySeqFrameProcessor(garment_name)
    finally:
        fullbody_seq.make_pix2pix_model = original_make_fn
    return processor


def process_video(video_path: str, garment_name: str, ckpt_dir: str, output_path: str):
    with _patched_overlay():
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
