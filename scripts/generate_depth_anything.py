import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_scenes(data_root, scene):
    data_root = Path(data_root)
    if scene:
        yield data_root / scene
        return
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and (path / "images").is_dir():
            yield path


def normalize_01(depth, eps=1e-6):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if valid.sum() == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo = np.percentile(depth[valid], 2.0)
    hi = np.percentile(depth[valid], 98.0)
    depth = np.clip(depth, lo, hi)
    return ((depth - lo) / max(hi - lo, eps)).astype(np.float32)


def predict_depth(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(processor, "post_process_depth_estimation"):
        depth = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )[0]["predicted_depth"]
    else:
        depth = outputs.predicted_depth
        if depth.dim() == 3:
            depth = depth[:, None]
        depth = F.interpolate(
            depth,
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
    if depth.shape[-2:] != (image.height, image.width):
        depth = F.interpolate(
            depth[None, None],
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
    return depth.detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--output-dir", default="mono_depth")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[DepthAnything] CUDA unavailable, using CPU")
        device = "cpu"

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device)
    model.eval()

    for scene_dir in iter_scenes(args.data_root, args.scene):
        image_dir = scene_dir / args.images_dir
        if not image_dir.is_dir():
            print(f"[DepthAnything] missing images dir: {image_dir}")
            continue
        out_dir = scene_dir / args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [
            p for p in sorted(image_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTS
        ]
        print(f"[DepthAnything] {scene_dir.name}: {len(image_paths)} images")
        for image_path in tqdm(image_paths, dynamic_ncols=True):
            out_path = out_dir / f"{image_path.stem}.npy"
            if out_path.exists() and not args.overwrite:
                continue
            image = Image.open(image_path).convert("RGB")
            depth = normalize_01(predict_depth(image, processor, model, device))
            if args.invert:
                depth = 1.0 - depth
            np.save(out_path, depth.astype(np.float32))


if __name__ == "__main__":
    main()
