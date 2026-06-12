"""
utils/viser_viewer.py — Live training viewer for PhysDecouple-4DGS.

Runs a viser web server in a background thread.  During training, call
    viewer.update(iteration, render_pkg, gt_image)
every N iterations to push the latest renders to the browser.

Open http://<server-ip>:7007 in any browser (no display server needed).

Panels (side-by-side in the GUI):
    GT         — ground-truth frame
    I_degraded — underwater render (what the model generates vs GT)
    clean_J    — restored clean scene
    backscatter — backscatter component
    sigma      — per-pixel transient score heatmap (bright = transient)
    depth      — normalised depth map
"""

import threading
from typing import Optional
import numpy as np
import torch

try:
    import viser
    _HAS_VISER = True
except ImportError:
    _HAS_VISER = False


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW float tensor → HWC uint8 numpy array."""
    t = t.detach().clamp(0.0, 1.0).float()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)          # CHW → HWC
    if t.shape[-1] == 1:
        t = t.expand(-1, -1, 3)
    return (t.cpu().numpy() * 255).astype(np.uint8)


def _thumb(arr: np.ndarray, max_h: int = 288, max_w: int = 512) -> np.ndarray:
    """Downscale HWC array to fit within max_h × max_w, preserving aspect ratio."""
    h, w = arr.shape[:2]
    scale = min(max_h / max(h, 1), max_w / max(w, 1), 1.0)
    if scale < 0.99:
        try:
            import cv2
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    return arr


def _resize_np(arr: np.ndarray, size_hw) -> np.ndarray:
    """Resize HWC uint8 image without making OpenCV a hard viewer dependency."""
    h, w = size_hw
    try:
        import cv2
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        try:
            from PIL import Image
            return np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR))
        except Exception:
            return arr[:h, :w]


class LiveViewer:
    """
    Viser-based live training viewer.

    Usage::

        viewer = LiveViewer(port=7007)
        # inside fine training loop every N iterations:
        viewer.update(iteration, render_pkg, gt_image)
    """

    def __init__(self, port: int = 7007, update_every: int = 200):
        self.update_every = update_every
        self._server = None
        self._handles = {}   # key → viser GuiImage handle
        self._lock = threading.Lock()

        if not _HAS_VISER:
            print("[Viewer] viser not installed — live viewer disabled.")
            return

        try:
            server = viser.ViserServer(host="0.0.0.0", port=port)
            self._server = server

            placeholder = np.zeros((2, 2, 3), dtype=np.uint8)

            with server.gui.add_folder("PhysDecouple-4DGS  Live"):
                self._iter_text = server.gui.add_text(
                    "Iteration", initial_value="—")

            # Two rows of three panels each
            panels = [
                ("0_GT | I_deg | J",  "compare"),
                ("1_GT",              "gt"),
                ("2_I_degraded",      "I_deg"),
                ("3_clean_J",         "J"),
                ("4_backscatter",     "backscatter"),
                ("5_sigma",           "sigma"),
                ("6_depth",           "depth"),
            ]
            for label, key in panels:
                with server.gui.add_folder(label):
                    self._handles[key] = server.gui.add_image(placeholder)

            print(f"[Viewer] Live viewer → http://0.0.0.0:{port}  "
                  "(open in any browser on your network)")

        except Exception as exc:
            print(f"[Viewer] Could not start: {exc}")
            self._server = None

    # ------------------------------------------------------------------

    def update(self, iteration: int, render_pkg: dict,
               gt_image: Optional[torch.Tensor] = None):
        """Push latest renders to browser. Safe to call every iteration."""
        if self._server is None:
            return
        if iteration % self.update_every != 0:
            return

        with self._lock:
            try:
                self._iter_text.value = f"iter {iteration:,}"

                def _push(key: str, t: Optional[torch.Tensor]):
                    if t is None or key not in self._handles:
                        return
                    arr = _thumb(_to_uint8(t))
                    self._handles[key].image = arr

                # NB: `a or b` on tensors raises (ambiguous bool) — always use
                # explicit None checks here.
                img_deg = render_pkg.get("render_degraded")
                if img_deg is None:
                    img_deg = render_pkg.get("render")   # coarse stage
                img_J = render_pkg.get("render_J")
                if img_J is None:
                    img_J = render_pkg.get("render_clean")

                _push("gt",          gt_image)
                _push("I_deg",       img_deg)
                _push("J",           img_J)
                _push("backscatter", render_pkg.get("render_backscatter"))

                sigma = render_pkg.get("sigma_render")
                if sigma is not None:
                    if sigma.shape[0] == 1:
                        sigma = sigma.expand(3, -1, -1)
                    _push("sigma", sigma)

                depth = render_pkg.get("depth_norm")
                if depth is not None:
                    if depth.dim() == 2:
                        depth = depth.unsqueeze(0).expand(3, -1, -1)
                    _push("depth", depth)

                # Side-by-side comparison strip  GT | I_deg | J
                parts = []
                for t in [gt_image, img_deg, img_J]:
                    if t is not None:
                        parts.append(_to_uint8(t))
                if len(parts) >= 2:
                    h = min(p.shape[0] for p in parts)
                    w = min(p.shape[1] for p in parts)
                    strip = np.concatenate(
                        [_resize_np(p, (h, w)) for p in parts], axis=1)
                    self._handles["compare"].image = _thumb(
                        strip, max_h=288, max_w=1536)

            except Exception as exc:
                # Never crash training, but don't hide bugs either.
                if not getattr(self, "_warned", False):
                    print(f"[Viewer] update failed (suppressed from now on): {exc}")
                    self._warned = True

    def close(self):
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
