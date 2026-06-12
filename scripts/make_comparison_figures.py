"""Generate paper-style qualitative comparison figures.

For each scene: one row per task (NVS reconstruction / UIR restoration),
columns = [GT, 6 SOTA methods, Ours], red+yellow crop boxes drawn on the
full frame and the two crops shown zoomed beneath each column.

Views are aligned across methods by matching each method's own GT (or
reconstruction) frame to ours with a downsampled L2 search, since every
method uses its own test-frame indexing.

Usage:
    python scripts/make_comparison_figures.py --scenes set4 set5 set8 \
        --ours-version v15 --out assets/results
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOTA = "/mnt/md0/IITM/ipcv/saiteja/SOTA"
OURS = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/output"

# method name -> (gt_dir_tpl, recon_dir_tpl, restore_dir_tpl); {s}=scene
METHODS = [
    ("SeaThru-NeRF",
     SOTA + "/seathru_nerf/{s}/{s}_uw_seathru_nerf/render/test_preds_step_250000/gt",
     SOTA + "/seathru_nerf/{s}/{s}_uw_seathru_nerf/render/test_preds_step_250000/color",
     SOTA + "/seathru_nerf/{s}/{s}_uw_seathru_nerf/render/test_preds_step_250000/restored"),
    ("WaterSplatting",
     SOTA + "/water-splatting/{s}/test/gt-rgb",
     SOTA + "/water-splatting/{s}/test/pred_image",
     SOTA + "/water-splatting/{s}/test/rgb_clear_clamp"),
    ("SeaSplat",
     SOTA + "/seasplat/{s}/test/with_water",
     SOTA + "/seasplat/{s}/test/with_water",
     SOTA + "/seasplat/{s}/test/no_water"),
    ("RUSplatting",
     SOTA + "/RUSplatting/{s}_adaptive_eval/test/ours_15000/gt",
     SOTA + "/RUSplatting/{s}_adaptive_eval/test/ours_15000/renders",
     SOTA + "/RUSplatting/{s}_adaptive_eval/test/ours_15000/clean"),
    ("UW-GS",
     SOTA + "/UW-GS/{s}/test/ours_15000/gt",
     SOTA + "/UW-GS/{s}/test/ours_15000/renders",
     SOTA + "/UW-GS/{s}/test/ours_15000/clean"),
    ("Plenodium",
     SOTA + "/plenodium/{s}_with_depth/test/gt-rgb",
     SOTA + "/plenodium/{s}_with_depth/test/pred_image",
     SOTA + "/plenodium/{s}_with_depth/test/rgb_clear_clamp"),
]

# per-scene chosen frame (index into OUR test set) and two crop boxes as
# fractions (x, y, w, h): red = far-field/medium region, yellow = detail.
SCENE_CFG = {
    "set4":  dict(frame=20, red=(0.62, 0.05, 0.22, 0.22), yellow=(0.30, 0.45, 0.22, 0.22)),
    "set5":  dict(frame=17, red=(0.66, 0.06, 0.22, 0.22), yellow=(0.38, 0.42, 0.22, 0.22)),
    "set8":  dict(frame=16, red=(0.60, 0.04, 0.22, 0.22), yellow=(0.18, 0.35, 0.22, 0.22)),
    "set9":  dict(frame=15, red=(0.62, 0.06, 0.22, 0.22), yellow=(0.35, 0.45, 0.22, 0.22)),
    "set12": dict(frame=10, red=(0.62, 0.06, 0.22, 0.22), yellow=(0.35, 0.45, 0.22, 0.22)),
}

COL_W = 480
LABEL_H = 26


def _imgs(d):
    if not d or not os.path.isdir(d):
        return []
    fs = sorted(glob.glob(os.path.join(d, "*.png"))
                + glob.glob(os.path.join(d, "*.jpg")))
    return fs


def _load(p, size=None):
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im


def _thumb_arr(p, hw=(18, 32)):
    im = Image.open(p).convert("L").resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def match_index(our_gt_path, cand_paths):
    """Index of cand whose downsampled gray image best matches ours."""
    ref = _thumb_arr(our_gt_path)
    best, bi = 1e9, None
    for i, p in enumerate(cand_paths):
        try:
            d = float(((ref - _thumb_arr(p)) ** 2).mean())
        except Exception:
            continue
        if d < best:
            best, bi = d, i
    return bi, best


def _font(sz=18):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(cand):
            return ImageFont.truetype(cand, sz)
    return ImageFont.load_default()


def column(im, red, yellow, label):
    """Full frame with boxes + two zoomed crops beneath + label."""
    W = COL_W
    H = int(im.height * W / im.width)
    full = im.resize((W, H), Image.LANCZOS)
    draw = ImageDraw.Draw(full)

    def box_px(b):
        return (int(b[0] * W), int(b[1] * H),
                int((b[0] + b[2]) * W), int((b[1] + b[3]) * H))

    rb, yb = box_px(red), box_px(yellow)
    draw.rectangle(rb, outline=(255, 40, 40), width=4)
    draw.rectangle(yb, outline=(255, 220, 40), width=4)
    draw.text((8, H - LABEL_H), label, fill="white", font=_font(18),
              stroke_width=2, stroke_fill="black")

    cw = W // 2
    crops = []
    for b, color in ((yb, (255, 220, 40)), (rb, (255, 40, 40))):
        c = full.crop(b).resize((cw, cw), Image.LANCZOS)
        d2 = ImageDraw.Draw(c)
        d2.rectangle([0, 0, cw - 1, cw - 1], outline=color, width=5)
        crops.append(c)
    strip = Image.new("RGB", (W, cw))
    strip.paste(crops[0], (0, 0))
    strip.paste(crops[1], (cw, 0))

    out = Image.new("RGB", (W, H + cw + 4), "white")
    out.paste(full, (0, 0))
    out.paste(strip, (0, H + 4))
    return out


def build_row(scene, task, ours_version, out_dir):
    cfg = SCENE_CFG[scene]
    base = f"{OURS}/{scene}_physdecouple_{ours_version}/test/ours_25000"
    our_gt = sorted(glob.glob(f"{base}/gt/*.png"))
    if not our_gt:
        print(f"[skip] {scene}: no our renders at {base}")
        return None
    fidx = min(cfg["frame"], len(our_gt) - 1)
    ref_gt = our_gt[fidx]

    ours_dir = (f"{base}/renders" if task == "nvs"
                else f"{base}/renders_clean_transient_removed")
    our_img = sorted(glob.glob(f"{ours_dir}/*.png"))[fidx]

    cols = [( "Groundtruth", ref_gt )]
    for name, gt_t, recon_t, restore_t in METHODS:
        cand_gt = _imgs(gt_t.format(s=scene))
        src_dir = (recon_t if task == "nvs" else restore_t).format(s=scene)
        cand_src = _imgs(src_dir)
        if not cand_gt or not cand_src:
            print(f"  [{scene}/{task}] {name}: missing ({src_dir})")
            continue
        mi, err = match_index(ref_gt, cand_gt)
        if mi is None or mi >= len(cand_src):
            mi = min(mi or 0, len(cand_src) - 1)
        cols.append((name, cand_src[mi]))
    cols.append(("Ours", our_img))

    ref_im = _load(ref_gt)
    rendered = [column(_load(p, ref_im.size), cfg["red"], cfg["yellow"], lab)
                for lab, p in cols]
    h = max(r.height for r in rendered)
    W = sum(r.width + 6 for r in rendered) - 6
    sheet = Image.new("RGB", (W, h), "white")
    x = 0
    for r in rendered:
        sheet.paste(r, (x, 0))
        x += r.width + 6
    out = os.path.join(out_dir, f"{scene}_{task}.png")
    sheet.save(out, optimize=True)
    print(f"[ok] {out}  cols={len(cols)}")
    return out


DECOMP = [
    ("Groundtruth", "gt"),
    ("Reconstruction $I$", "renders"),
    ("Restoration $J$", "renders_clean"),
    ("$J$, transients removed", "renders_clean_transient_removed"),
    ("Backscatter", "backscatter"),
    ("Transient map $\\sigma$", "sigma"),
    ("Depth", "depth_norm"),
]


def build_decomposition(scene, ours_version, out_dir, n_frames=2):
    """Full model factorization — the outputs only our model provides."""
    cfg = SCENE_CFG[scene]
    base = f"{OURS}/{scene}_physdecouple_{ours_version}/test/ours_25000"
    gts = sorted(glob.glob(f"{base}/gt/*.png"))
    if not gts:
        print(f"[skip-decomp] {scene}")
        return None
    n = len(gts)
    frames = [min(cfg["frame"], n - 1), (cfg["frame"] + n // 2) % n][:n_frames]
    rows = []
    for fi in frames:
        cols = []
        for label, sub in DECOMP:
            fs = sorted(glob.glob(f"{base}/{sub}/*.png"))
            if not fs:
                continue
            im = _load(fs[min(fi, len(fs) - 1)])
            if sub == "backscatter":
                # display-normalized (x3) — values are physically small
                arr = np.asarray(im, dtype=np.float32) * 3.0
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            W = 360
            H = int(im.height * W / im.width)
            im = im.resize((W, H), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            plain = label.replace("$", "").replace("\\sigma", "sigma")
            d.text((6, H - 24), plain, fill="white", font=_font(15),
                   stroke_width=2, stroke_fill="black")
            cols.append(im)
        if cols:
            h = max(c.height for c in cols)
            W = sum(c.width + 4 for c in cols) - 4
            row = Image.new("RGB", (W, h), "white")
            x = 0
            for c in cols:
                row.paste(c, (x, 0))
                x += c.width + 4
            rows.append(row)
    if not rows:
        return None
    W = max(r.width for r in rows)
    H = sum(r.height + 6 for r in rows) - 6
    sheet = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    out = os.path.join(out_dir, f"{scene}_decomposition.png")
    sheet.save(out, optimize=True)
    print(f"[ok] {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["set4", "set5", "set8"])
    ap.add_argument("--ours-version", default="v15")
    ap.add_argument("--out", default="assets/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for s in args.scenes:
        for task in ("uir", "nvs"):
            build_row(s, task, args.ours_version, args.out)
        build_decomposition(s, args.ours_version, args.out)


if __name__ == "__main__":
    main()
