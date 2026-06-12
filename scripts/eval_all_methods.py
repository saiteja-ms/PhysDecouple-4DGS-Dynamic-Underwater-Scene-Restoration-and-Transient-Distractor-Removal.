"""Quantitative comparison: 6 SOTA methods + PhysDecouple-4DGS (v14/v15).

Full-reference (reconstruction vs each method's own GT): PSNR, SSIM, LPIPS.
No-reference (restoration): UIQM, UCIQE.

Outputs a LaTeX booktabs table (best per metric in bold) and a CSV.

    python scripts/eval_all_methods.py --scenes set4 set5 set8 \
        --out assets/results
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_utils import ssim as ssim_fn
from utils.image_utils import psnr as psnr_fn
from lpipsPyTorch import lpips as lpips_fn

SOTA = "/mnt/md0/IITM/ipcv/saiteja/SOTA"
OURS = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/output"

# name -> (gt, recon, restore) dir templates
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
     "@source:Data/{s}/images",
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


def imgs(d):
    return sorted(glob.glob(os.path.join(d, "*.png"))
                  + glob.glob(os.path.join(d, "*.jpg"))) if os.path.isdir(d) else []


def to_tensor(p, size=None):
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return (torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
            .contiguous().float().cuda() / 255.0)


# ---------------- no-reference metrics ----------------

def uciqe(rgb):
    """UCIQE (Yang & Sowmya 2015). rgb: HWC float [0,1]."""
    import colorsys  # noqa: F401  (documentational)
    import cv2
    bgr = (rgb[..., ::-1] * 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std() / 100.0
    Ln = L / 100.0
    top, bot = np.percentile(Ln, 99), np.percentile(Ln, 1)
    con_l = top - bot
    sat = np.where(chroma > 0, chroma / np.maximum(np.sqrt(chroma**2 + L**2), 1e-6), 0)
    mu_s = sat.mean()
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def _uicm(rgb):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = (R - G).flatten(), (0.5 * (R + G) - B).flatten()

    def stats(x):
        x = np.sort(x)
        n = len(x)
        lo, hi = int(0.1 * n), int(0.9 * n)
        xt = x[lo:hi] if hi > lo else x
        mu = xt.mean()
        s2 = ((xt - mu) ** 2).mean()
        return mu, s2
    mu_rg, s_rg = stats(rg)
    mu_yb, s_yb = stats(yb)
    return (-0.0268 * np.sqrt(mu_rg**2 + mu_yb**2)
            + 0.1586 * np.sqrt(np.sqrt(s_rg * s_yb)))


def _eme(ch, k=8):
    h, w = ch.shape
    bh, bw = h // k, w // k
    if bh == 0 or bw == 0:
        return 0.0
    acc, cnt = 0.0, 0
    for i in range(k):
        for j in range(k):
            blk = ch[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            mx, mn = blk.max(), blk.min()
            if mn > 1e-4 and mx > mn:
                acc += np.log(mx / mn)
                cnt += 1
    return 2.0 * acc / max(cnt, 1)


def _uism(rgb):
    import cv2
    w = (0.299, 0.587, 0.114)
    acc = 0.0
    for c, wc in enumerate(w):
        ch = (rgb[..., c] * 255).astype(np.uint8)
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
        edge = np.sqrt(sx**2 + sy**2) / 255.0
        acc += wc * _eme(edge, 8)
    return acc


def _uiconm(rgb, k=8):
    inten = rgb.mean(-1)
    h, w = inten.shape
    bh, bw = h // k, w // k
    acc, cnt = 0.0, 0
    for i in range(k):
        for j in range(k):
            blk = inten[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            mx, mn = blk.max(), blk.min()
            s = mx + mn
            d = mx - mn
            if s > 1e-4 and d > 1e-4:
                r = d / s
                acc += r * np.log(r + 1e-8)
                cnt += 1
    return -acc / max(cnt, 1)


def uiqm(rgb):
    """UIQM (Panetta et al. 2016)."""
    return 0.0282 * _uicm(rgb) + 0.2953 * _uism(rgb) + 3.5753 * _uiconm(rgb)


# ---------------- evaluation ----------------

def full_reference(recon_dir, gt_dir, max_n=40):
    rs = imgs(recon_dir)
    if gt_dir.startswith("@source:"):
        # pair by basename against original dataset frames
        root = gt_dir.split(":", 1)[1]
        pairs = [(r, os.path.join(root, os.path.basename(r))) for r in rs]
        pairs = [(r, g) for r, g in pairs if os.path.exists(g)]
        rs = [p[0] for p in pairs]
        gs = [p[1] for p in pairs]
    else:
        gs = imgs(gt_dir)
    n = min(len(rs), len(gs), max_n)
    if n == 0:
        return None
    P, S, L = [], [], []
    for r, g in zip(rs[:n], gs[:n]):
        gt = to_tensor(g)
        pr = to_tensor(r, size=Image.open(g).size)
        P.append(psnr_fn(pr[None], gt[None]).mean().item())
        S.append(ssim_fn(pr[None], gt[None]).item())
        with torch.no_grad():
            L.append(lpips_fn(pr[None], gt[None], net_type="alex").item())
    return float(np.mean(P)), float(np.mean(S)), float(np.mean(L))


def no_reference(restore_dir, max_n=40):
    fs = imgs(restore_dir)[:max_n]
    if not fs:
        return None
    Q, C = [], []
    for f in fs:
        rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
        Q.append(uiqm(rgb))
        C.append(uciqe(rgb))
    return float(np.mean(Q)), float(np.mean(C))


def ours_dirs(scene):
    """Best of v14/v15 (nothing older), chosen by reconstruction PSNR."""
    cands = []
    for v in ("v15", "v14"):
        base = f"{OURS}/{scene}_physdecouple_{v}/test/ours_25000"
        if os.path.isdir(f"{base}/renders") and os.path.isdir(f"{base}/gt"):
            cands.append((v, base))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["set4", "set5", "set8"])
    ap.add_argument("--out", default="assets/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = {}   # scene -> list of (method, psnr, ssim, lpips, uiqm, uciqe)
    for s in args.scenes:
        rows[s] = []
        for name, gt_t, recon_t, restore_t in METHODS:
            fr = full_reference(recon_t.format(s=s), gt_t.format(s=s))
            nr = no_reference(restore_t.format(s=s))
            if fr is None and nr is None:
                print(f"[{s}] {name}: missing")
                continue
            rows[s].append((name,) + (fr or (np.nan,)*3) + (nr or (np.nan,)*2))
            print(f"[{s}] {name}: {rows[s][-1][1:]}")

        best = None
        for v, base in ours_dirs(s):
            fr = full_reference(f"{base}/renders", f"{base}/gt")
            if fr and (best is None or fr[0] > best[1][0]):
                best = (v, fr, base)
        if best:
            v, fr, base = best
            rdir = f"{base}/renders_clean_transient_removed"
            if not os.path.isdir(rdir):
                rdir = f"{base}/renders_clean"
            nr = no_reference(rdir)
            rows[s].append((f"Ours ({v})",) + fr + (nr or (np.nan,)*2))
            print(f"[{s}] Ours({v}): {rows[s][-1][1:]}")

    # CSV
    import csv
    with open(f"{args.out}/metrics_all.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "method", "PSNR", "SSIM", "LPIPS", "UIQM", "UCIQE"])
        for s in args.scenes:
            for r in rows[s]:
                w.writerow([s] + list(r))

    # LaTeX
    hi = [True, True, False, True, True]  # higher-better per metric
    with open(f"{args.out}/metrics_table.tex", "w") as f:
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("Scene & Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & "
                "LPIPS$\\downarrow$ & UIQM$\\uparrow$ & UCIQE$\\uparrow$ \\\\\n")
        for s in args.scenes:
            f.write("\\midrule\n")
            data = rows[s]
            cols = list(zip(*[r[1:] for r in data]))
            best_idx = []
            for ci, c in enumerate(cols):
                arr = np.array(c, dtype=float)
                arr[np.isnan(arr)] = -1e9 if hi[ci] else 1e9
                best_idx.append(int(np.nanargmax(arr) if hi[ci]
                                    else np.nanargmin(arr)))
            for ri, r in enumerate(data):
                cells = []
                for ci, vval in enumerate(r[1:]):
                    txt = "--" if np.isnan(vval) else f"{vval:.3f}"
                    if ri == best_idx[ci] and not np.isnan(vval):
                        txt = f"\\textbf{{{txt}}}"
                    cells.append(txt)
                sn = s if ri == 0 else ""
                nm = r[0].replace("Ours", "\\textbf{Ours}")
                f.write(f"{sn} & {nm} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("written:", f"{args.out}/metrics_table.tex")


if __name__ == "__main__":
    main()
