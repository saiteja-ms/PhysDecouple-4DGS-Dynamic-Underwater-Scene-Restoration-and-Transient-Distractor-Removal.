import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEPTH_DIR_CANDIDATES = ("mono_depth", "depth_anything")
DEPTH_EXT_CANDIDATES = (".npy", ".npz", ".png", ".tiff", ".tif")


def _scene_root_from_image(image_path):
    image_path = Path(image_path)
    if image_path.parent.name in ("images", "ims"):
        return image_path.parent.parent
    return image_path.parent


def find_mono_depth_path(image_path, depth_root=None):
    image_path = Path(image_path)
    stem = image_path.stem

    roots = []
    if depth_root is not None:
        roots.append(Path(depth_root))
    scene_root = _scene_root_from_image(image_path)
    roots.extend(scene_root / name for name in DEPTH_DIR_CANDIDATES)

    for root in roots:
        for ext in DEPTH_EXT_CANDIDATES:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)
    return None


def _read_depth_file(path):
    path = Path(path)
    if path.suffix == ".npy":
        depth = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = data[key]
    else:
        depth = np.asarray(Image.open(path))
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def normalize_depth_01(depth, eps=1e-6):
    valid = np.isfinite(depth)
    if valid.sum() == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo = np.percentile(depth[valid], 2.0)
    hi = np.percentile(depth[valid], 98.0)
    depth = np.clip(depth, lo, hi)
    return ((depth - lo) / max(hi - lo, eps)).astype(np.float32)


def load_mono_depth(image_path, target_hw=None, depth_root=None):
    depth_path = find_mono_depth_path(image_path, depth_root=depth_root)
    if depth_path is None:
        return None
    depth = normalize_depth_01(_read_depth_file(depth_path))
    depth_t = torch.from_numpy(depth).float()
    if target_hw is not None and tuple(depth_t.shape[-2:]) != tuple(target_hw):
        depth_t = F.interpolate(
            depth_t[None, None],
            size=tuple(target_hw),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return depth_t


def robust_normalize_depth_torch(depth, mask=None, eps=1e-6):
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    depth = depth.float()
    valid = torch.isfinite(depth)
    if mask is not None:
        valid = valid & mask.bool()
    values = depth[valid]
    if values.numel() < 16:
        return None, valid
    median = values.median()
    mad = (values - median).abs().median().clamp_min(eps)
    return (depth - median) / mad, valid


def scale_shift_invariant_depth_loss(depth_pred, depth_target, mask=None,
                                     residual_clip=2.0):
    pred_n, valid = robust_normalize_depth_torch(depth_pred, mask=mask)
    target_n, valid_t = robust_normalize_depth_torch(depth_target, mask=valid)
    if pred_n is None or target_n is None:
        return torch.zeros((), device=depth_pred.device, dtype=depth_pred.dtype)
    valid = valid & valid_t
    if valid.sum() < 16:
        return torch.zeros((), device=depth_pred.device, dtype=depth_pred.dtype)
    residual = (pred_n[valid] - target_n[valid].detach()).abs()
    if residual_clip is not None and residual_clip > 0:
        residual = residual.clamp(max=float(residual_clip))
    return residual.mean()


def pearson_depth_loss(depth_pred, depth_target, mask=None, eps=1e-6):
    """
    Scale/shift invariant monocular depth supervision.

    Depth-Anything-style predictions are relative depths, so Pearson correlation
    is the right signal: match ordering/shape, not metric scale or offset.
    """
    if depth_pred.dim() == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_target.dim() == 3:
        depth_target = depth_target.squeeze(0)

    valid = torch.isfinite(depth_pred) & torch.isfinite(depth_target)
    if mask is not None:
        valid = valid & mask.bool()
    if valid.sum() < 16:
        return torch.zeros((), device=depth_pred.device, dtype=depth_pred.dtype)

    pred = depth_pred[valid].float()
    target = depth_target[valid].float().detach()
    pred = pred - pred.mean()
    target = target - target.mean()
    corr = (pred * target).mean() / (
        pred.square().mean().sqrt() * target.square().mean().sqrt() + eps)
    return 1.0 - corr.clamp(-1.0, 1.0)
