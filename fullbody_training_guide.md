# Full-Body Garment Training Guide

This guide extends the existing upper-body pipeline to support full-body garments. It covers dataset preparation, training, and using the resulting checkpoint inside `demo.py`.

## Requirements
- Recorded video of a person wearing the target garment (4K capture recommended).
- Per-frame garment segmentation masks with an alpha channel.
- Installed dependencies specified in `requirements.txt` (the `rtv` conda environment from this repo works for both generation and training).

## Step 1 — Capture Reference Video
Record a video following the pose guidance from the upper-body instructions (diverse poses improve results). A sample is available in the original `training_instructions.md`.

## Step 2 — Generate Segmentation Masks
Use your preferred method (e.g., [muggled_sam](https://github.com/heyoeyo/muggled_sam)) to produce high-quality garment masks. Export them as PNG files with alpha channels, ordered to align with the video frames.

## Step 3 — Build the Full-Body Dataset
Extract the mask archive to a directory and run:

```bash
python DatasetGeneration/fullbody_dataset_generation.py \
  --video_path <path/to/video.mp4> \
  --mask_dir <path/to/mask_directory> \
  --dataset_name <garment_dataset_name>
```

The script stores warped garment crops, DensePose data, SMPL virtual mannequins, and affine transforms under `./PerGarmentDatasets/<garment_dataset_name>`. Metadata (`dataset_info.json`) captures the original frame size and the full-body ROI dimensions (1024×768).

## Step 4 — Train a Full-Body Checkpoint
Launch training with:

```bash
python Training/fullbody_training.py \
  --model pix2pixHD_RNN_RGBA \
  --input_nc 6 \
  --output_nc 4 \
  --batchSize 4 \
  --img_size 576 \
  --dataset_path ./PerGarmentDatasets/<garment_dataset_name> \
  --name <checkpoint_name> \
  --niter 80 \
  --niter_decay 80
```

- `img_size` controls the ROI height; the loader derives width automatically (producing 576×432 tensors to match the runtime pipeline).
- You can chain multiple datasets with `--dataset_path path_a,path_b`.
- Checkpoints and training logs appear under `./checkpoints/<checkpoint_name>`.

## Step 5 — Test in the Demo
Point the demo to your trained model by updating the checkpoint name passed to `FullBodySeqFrameProcessor` inside `demo.py`:

```python
self.frame_processor = FullBodySeqFrameProcessor('<checkpoint_name>')
```

Run the application inside the `rtv` environment:

```bash
python demo.py --camera-rotate -90  # adjust flag as needed for your camera orientation
```

When the checkpoint loads correctly, the live stream should display the new garment rendered on the person in view.

### Notes
- Ensure masks align with the video frames; mismatches will introduce artifacts during training.
- If you capture data at a different resolution, the dataset script automatically records the original frame size and handles reprojection.
- Training is GPU-intensive; monitor VRAM usage and reduce `batchSize` if needed.
