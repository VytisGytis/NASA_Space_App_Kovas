"""
Simple laser speckle contrast imaging (LSCI) pipeline.

Usage:
    python lsci_pipeline.py frames_folder [--win 7] [--roi x0 y0 x1 y1]

Put your captured frames (dng/png/tif/bmp/jpg) in one folder, sorted by name.
Requires: numpy scipy pillow matplotlib rawpy
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt


def load_dng(path):
    """
    Read a DNG as RAW Bayer data (no demosaicing) and return one green
    sample per 2x2 Bayer tile as a float array (H/2, W/2).

    Demosaicing interpolates neighbouring pixels, which correlates them and
    artificially lowers speckle contrast, so we avoid it on purpose.
    """
    import rawpy  # imported here so BMP/PNG users don't need it

    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(np.float64)
        colors = raw.raw_colors_visible          # 0=R, 1=G, 2=B, 3=G2
        black = float(np.mean(raw.black_level_per_channel))

    mosaic = np.clip(mosaic - black, 0, None)
    h, w = (mosaic.shape[0] // 2) * 2, (mosaic.shape[1] // 2) * 2
    mosaic, colors = mosaic[:h, :w], colors[:h, :w]

    green = (colors == 1)                         # exactly one G per 2x2 tile
    blocks = (mosaic * green).reshape(h // 2, 2, w // 2, 2)
    return blocks.sum(axis=(1, 3))


def load_image(path):
    """Load one frame as a 2D float array."""
    if path.suffix.lower() == ".dng":
        return load_dng(path)
    img = np.asarray(Image.open(path))
    if img.ndim == 3:          # RGB -> keep green channel
        img = img[..., 1]
    return img.astype(np.float64)


def load_frames(folder):
    """Load all images in a folder as a (T, H, W) float array."""
    exts = {".dng", ".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    frames = [load_image(f) for f in files]
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(f"Frames have different sizes: {shapes}")
    return np.stack(frames)


def spatial_contrast(img, win=7):
    """K = std/mean in a win x win sliding window (fast, vectorized)."""
    mean = uniform_filter(img, win)
    mean_sq = uniform_filter(img ** 2, win)
    var = np.clip(mean_sq - mean ** 2, 0, None)
    return np.sqrt(var) / (mean + 1e-9)


def temporal_contrast(stack):
    """K = std/mean along the time axis, per pixel."""
    return stack.std(axis=0) / (stack.mean(axis=0) + 1e-9)


def flow_index(K):
    """Relative flow index 1/K^2 (NOT absolute blood flow)."""
    return 1.0 / (K ** 2 + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--win", type=int, default=7, help="odd window size")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--out", default="lsci_out")
    args = ap.parse_args()

    stack = load_frames(args.folder)
    print(f"Loaded {stack.shape[0]} frames of {stack.shape[1]}x{stack.shape[2]}")

    # Spatial contrast per frame, then average (reduces noise)
    K_spatial = np.mean([spatial_contrast(f, args.win) for f in stack], axis=0)
    K_temporal = temporal_contrast(stack) if len(stack) > 1 else None

    K = K_spatial
    flow = flow_index(K)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    np.save(out / "K_spatial.npy", K_spatial)

    fig, axes = plt.subplots(1, 3 if K_temporal is not None else 2,
                             figsize=(14, 5))
    im0 = axes[0].imshow(K, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Spatial contrast K")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    hi = np.percentile(flow, 99)
    im1 = axes[1].imshow(flow, cmap="hot", vmin=0, vmax=hi)
    axes[1].set_title("Relative flow index (1/K²)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    if K_temporal is not None:
        im2 = axes[2].imshow(K_temporal, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Temporal contrast K")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")

    if args.roi:
        x0, y0, x1, y1 = args.roi
        print(f"ROI mean K = {K[y0:y1, x0:x1].mean():.4f}, "
              f"mean 1/K² = {flow[y0:y1, x0:x1].mean():.2f}")

    plt.tight_layout()
    plt.savefig(out / "lsci_result.png", dpi=150)
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()