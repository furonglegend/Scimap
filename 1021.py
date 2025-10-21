# demo_scimap_he_fusion_diagnostic_multi.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import anndata as ad
from skimage import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- User inputs ----------
COUNTS_CSV = r"/home/ubuntu-user/kkk/102101/counts_matrix.csv"
META_CSV   = r"/home/ubuntu-user/kkk/102101/meta_data_with_X_Y.csv"
HE_IMAGE_PATH = r"/home/ubuntu-user/kkk/102101/HE_image01.tif"

# choose markers to produce single-marker overlays (list)
MARKERS_TO_PLOT = ["CD3", "CD8", "PanCK"]   
# choose a triple for RGB composite (exactly 3 names, or None to skip)
COMPOSITE_TRIPLE = ("PanCK", "CD3", "CD8")  # (R, G, B) -> change as desired or set to None

# other settings
X_COL = "X_centroid"
Y_COL = "Y_centroid"
OUT_DIR = "./he_overlay_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RADIUS_PIXELS = 50.0  # used for informational sizing; scatter size derived from marker
FLIP_Y = False        # True
VERBOSE = True

# ---------- small utils ----------
def safe_to_numpy(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()

def load_marker(counts_df, marker):
    if marker in counts_df.columns:
        return safe_to_numpy(counts_df[marker].values.astype(float))
    else:
        # missing -> zeros and warn
        print(f"[WARN] marker '{marker}' not found in counts; filling zeros.")
        return np.zeros(len(counts_df), dtype=float)

def normalize_0_1(arr):
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn <= 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# ---------- Load data ----------
counts = pd.read_csv(COUNTS_CSV, index_col=0)
meta   = pd.read_csv(META_CSV, index_col=0)

# ensure same ordering
counts = counts.loc[meta.index]

adata = ad.AnnData(X=counts.values)
adata.obs = meta.copy()
adata.var_names = counts.columns
adata.obs_names = counts.index.astype(str)

# ---------- Load HE image ----------
try:
    img = io.imread(HE_IMAGE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed reading HE image at {HE_IMAGE_PATH}: {e}")

if img.ndim == 2:
    rgb = np.stack([img, img, img], axis=-1)
else:
    rgb = img

h, w = rgb.shape[0], rgb.shape[1]
if VERBOSE:
    print("Image shape:", rgb.shape, "dtype:", rgb.dtype)

# ---------- Coordinates ----------
xs = adata.obs[X_COL].astype(float).values
ys = adata.obs[Y_COL].astype(float).values
if FLIP_Y:
    ys = h - ys

# scaling heuristics (same as原脚本)
scale_x = w / (xs.max() if xs.max()>0 else 1)
scale_y = h / (ys.max() if ys.max()>0 else 1)
scale = min(scale_x, scale_y)

# helper to save three standard views for a marker
def plot_marker_views(marker, marker_vals_raw):
    mv = normalize_0_1(marker_vals_raw)
    sz = 6 + 10 * mv  # visual size
    # 1) extent mapping (map axes to image pixel coords 0..w,0..h)
    out_extent = os.path.join(OUT_DIR, f"{marker}_extent.png")
    plt.figure(figsize=(8,8))
    plt.imshow(rgb, origin='upper', interpolation='nearest', extent=(0, w, h, 0))
    plt.scatter(xs, ys, c=mv, s=sz, cmap='viridis', alpha=0.7, linewidths=0)
    plt.title(f"{marker} overlay (extent)")
    plt.axis('off')
    plt.colorbar(label=f"{marker} (norm)")
    plt.savefig(out_extent, dpi=200, bbox_inches='tight')
    plt.close()
    if VERBOSE: print("Saved", out_extent)

    # 2) scaled coords
    out_scaled = os.path.join(OUT_DIR, f"{marker}_scaled.png")
    xs_scaled = xs * scale
    ys_scaled = ys * scale
    plt.figure(figsize=(8,8))
    plt.imshow(rgb, origin='upper', interpolation='nearest')
    plt.scatter(xs_scaled, ys_scaled, c=mv, s=sz, cmap='viridis', alpha=0.7, linewidths=0)
    plt.title(f"{marker} overlay (scaled by {scale:.4f})")
    plt.axis('off')
    plt.colorbar(label=f"{marker} (norm)")
    plt.savefig(out_scaled, dpi=200, bbox_inches='tight')
    plt.close()
    if VERBOSE: print("Saved", out_scaled)

    # 3) bbox overlay (no extent/no scaling) with bbox rectangle
    out_bbox = os.path.join(OUT_DIR, f"{marker}_bbox.png")
    bbox_x0, bbox_x1 = float(xs.min()), float(xs.max())
    bbox_y0, bbox_y1 = float(ys.min()), float(ys.max())
    plt.figure(figsize=(8,8))
    plt.imshow(rgb, origin='upper', interpolation='nearest')
    bx0 = max(0, bbox_x0); by0 = max(0, bbox_y0)
    bx1 = min(w, bbox_x1); by1 = min(h, bbox_y1)
    rect_x = [bx0, bx1, bx1, bx0, bx0]
    rect_y = [by0, by0, by1, by1, by0]
    plt.plot(rect_x, rect_y, color='red', linewidth=2)
    plt.title(f"{marker} bbox (red)")
    plt.axis('off')
    plt.savefig(out_bbox, dpi=200, bbox_inches='tight')
    plt.close()
    if VERBOSE: print("Saved", out_bbox)

# ---------- Plot single/multiple markers ----------
for marker in MARKERS_TO_PLOT:
    vals = load_marker(counts, marker)
    plot_marker_views(marker, vals)

# ---------- Plot multiple markers on same image (different symbols/colors) ----------
if len(MARKERS_TO_PLOT) > 1:
    # draw each marker as a different scatter style on top of the same image (extent)
    out_multi = os.path.join(OUT_DIR, "multi_markers_extent.png")
    plt.figure(figsize=(10,10))
    plt.imshow(rgb, origin='upper', interpolation='nearest', extent=(0, w, h, 0))
    markers_styles = ['o', 's', 'D', '^', 'v', 'P', 'X']  # cycle if needed
    for i, m in enumerate(MARKERS_TO_PLOT):
        vals = load_marker(counts, m)
        mv = normalize_0_1(vals)
        sz = 20 * (0.6 + mv)  # bigger for visibility
        plt.scatter(xs, ys, c=mv, s=sz, cmap='viridis', alpha=0.6, linewidths=0,
                    marker=markers_styles[i % len(markers_styles)], label=m)
    plt.legend(loc='upper right', fontsize='small')
    plt.title("Multiple markers overlay (extent)")
    plt.axis('off')
    plt.savefig(out_multi, dpi=200, bbox_inches='tight')
    plt.close()
    if VERBOSE: print("Saved", out_multi)

# ---------- Composite RGB (if requested) ----------
if COMPOSITE_TRIPLE is not None:
    if len(COMPOSITE_TRIPLE) != 3:
        print("[WARN] COMPOSITE_TRIPLE must be length 3 or None. Skipping composite.")
    else:
        r_name, g_name, b_name = COMPOSITE_TRIPLE
        r = normalize_0_1(load_marker(counts, r_name))
        g = normalize_0_1(load_marker(counts, g_name))
        b = normalize_0_1(load_marker(counts, b_name))
        colors = np.stack([r, g, b], axis=1)
        sz = 6 + 10 * (r + g + b) / 3.0
        out_rgb = os.path.join(OUT_DIR, f"composite_RGB_{r_name}_{g_name}_{b_name}.png")
        plt.figure(figsize=(8,8))
        plt.imshow(rgb, origin='upper', interpolation='nearest')
        plt.scatter(xs, ys, c=colors, s=sz, alpha=0.8, linewidths=0)
        plt.title(f"RGB composite: R={r_name}, G={g_name}, B={b_name}")
        plt.axis('off')
        plt.savefig(out_rgb, dpi=200, bbox_inches='tight')
        plt.close()
        if VERBOSE: print("Saved", out_rgb)

print("All done. Outputs saved to:", OUT_DIR)

