import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Base paths
OURS = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/output"
OUT_DIR = "/mnt/md0/IITM/ipcv/saiteja/PhysDecouple-4DGS/assets/results"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def _font(sz=15):
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, sz)
    return ImageFont.load_default()

def _load(p):
    return Image.open(p).convert("RGB")

def build_row_decomp(base_dir, frame_idx, decomp_cfg, row_label=None):
    cols = []
    for label, sub in decomp_cfg:
        fs = sorted(glob.glob(os.path.join(base_dir, sub, "*.png")))
        if not fs:
            print(f"[WARN] No files found in {os.path.join(base_dir, sub)}")
            continue
        
        # Load frame
        fidx = min(frame_idx, len(fs) - 1)
        im = _load(fs[fidx])
        
        if sub == "backscatter":
            # display-normalized (x3) — values are physically small
            arr = np.asarray(im, dtype=np.float32) * 3.0
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            
        W = 360
        H = int(im.height * W / im.width)
        im = im.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        
        # Replace LaTeX notation for rendering plain text
        plain = label.replace("$", "").replace("\\sigma", "sigma")
        d.text((6, H - 24), plain, fill="white", font=_font(15),
               stroke_width=2, stroke_fill="black")
        cols.append(im)
        
    if not cols:
        return None
        
    # Stack columns horizontally
    h = max(c.height for c in cols)
    W = sum(c.width + 4 for c in cols) - 4
    
    # If a row label (like Curasao, composite) is requested, we can optionally prepend or overlay it.
    # Overlay is cleaner and keeps the widths aligned.
    if row_label:
        # We can write the row label on the Groundtruth column (first column) at the top-left
        d = ImageDraw.Draw(cols[0])
        d.text((6, 6), row_label, fill="yellow", font=_font(16),
               stroke_width=2, stroke_fill="black")
        
    row = Image.new("RGB", (W, h), "white")
    x = 0
    for c in cols:
        row.paste(c, (x, 0))
        x += c.width + 4
    return row

def generate_seathrunerf():
    # SeaThru-NeRF is static, so we don't have transient components (or they are empty)
    # We display 5 columns: Groundtruth, Reconstruction, Restoration, Backscatter, Depth
    DECOMP_STATIC = [
        ("Groundtruth", "gt"),
        ("Reconstruction $I$", "renders"),
        ("Restoration $J$", "renders_clean"),
        ("Backscatter", "backscatter"),
        ("Depth", "depth_norm"),
    ]
    
    scenes = [
        ("Curasao", "Curasao", 1),
        ("IUI3-RedSea", "IUI3-RedSea", 1),
        ("JapaneseGradens-RedSea", "JapaneseGradens-RedSea", 1),
        ("Panama", "Panama", 1),
    ]
    
    rows = []
    for label, folder, frame_idx in scenes:
        base = os.path.join(OURS, "seathrunerf", folder, "test", "ours_25000")
        row = build_row_decomp(base, frame_idx, DECOMP_STATIC, row_label=label)
        if row is not None:
            rows.append(row)
            
    if not rows:
        print("No rows generated for SeaThru-NeRF!")
        return
        
    W = max(r.width for r in rows)
    H = sum(r.height + 6 for r in rows) - 6
    sheet = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
        
    out_path = os.path.join(OUT_DIR, "seathrunerf_decomposition.png")
    sheet.save(out_path, optimize=True)
    print(f"[SUCCESS] Saved SeaThru-NeRF decomposition grid to {out_path}")

def generate_nusr():
    # NUSR is dynamic, so we include transient columns (7 columns total)
    DECOMP_DYNAMIC = [
        ("Groundtruth", "gt"),
        ("Reconstruction $I$", "renders"),
        ("Restoration $J$", "renders_clean"),
        ("$J$, transients removed", "renders_clean_transient_removed"),
        ("Backscatter", "backscatter"),
        ("Transient map $\\sigma$", "sigma"),
        ("Depth", "depth_norm"),
    ]
    
    scenes = [
        ("composite", "composite", "ours_16000", [2, 8]),
        ("sardine", "sardine", "ours_14000", [2, 6]),
    ]
    
    rows = []
    for label, folder, ours_iter, frame_indices in scenes:
        base = os.path.join(OURS, "nusr", folder, "test", ours_iter)
        for i, fidx in enumerate(frame_indices):
            row_label = f"{label} (t={fidx})"
            row = build_row_decomp(base, fidx, DECOMP_DYNAMIC, row_label=row_label)
            if row is not None:
                rows.append(row)
                
    if not rows:
        print("No rows generated for NUSR!")
        return
        
    W = max(r.width for r in rows)
    H = sum(r.height + 6 for r in rows) - 6
    sheet = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
        
    out_path = os.path.join(OUT_DIR, "nusr_decomposition.png")
    sheet.save(out_path, optimize=True)
    print(f"[SUCCESS] Saved NUSR decomposition grid to {out_path}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    generate_seathrunerf()
    generate_nusr()
