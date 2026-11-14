"""
Laser speckle processing: loading frames (DNG or ordinary images), speckle
contrast, and change detection between two scans.

Everything here is GUI-free so it can be reused from scripts / notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, uniform_filter

try:
    import rawpy
except ImportError:  # DNG support is optional, PNG/TIFF/JPG still work
    rawpy = None

RAW_EXT = {".dng"}
IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
CHANNELS = ("Auto", "R", "G", "B", "Luma")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Frame:
    path: Path
    data: np.ndarray          # 2-D float64, black level removed
    saturated: float          # fraction of pixels at the sensor's white level
    channel: str


def list_frames(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in RAW_EXT | IMG_EXT)


def _raw_planes(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return one native-resolution Bayer plane per colour (no demosaicing,
    so the speckle statistics are not smeared by interpolation)."""
    if rawpy is None:
        raise RuntimeError("rawpy is not installed - cannot read DNG files")
    with rawpy.imread(str(path)) as r:
        raw = r.raw_image_visible
        pat = r.raw_pattern
        desc = r.color_desc.decode()
        black = r.black_level_per_channel
        white = float(r.white_level)
        planes, sat = {}, {}
        for i in range(2):
            for j in range(2):
                idx = int(pat[i, j])
                letter = desc[idx]
                if letter in planes:          # keep only the first green
                    continue
                p = raw[i::2, j::2]
                sat[letter] = float((p >= white * 0.98).mean())
                planes[letter] = p.astype(np.float64) - float(black[idx])
        return planes, sat


def _img_planes(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    img = np.asarray(Image.open(path))
    top = 255.0 if img.dtype == np.uint8 else float(np.iinfo(img.dtype).max) \
        if np.issubdtype(img.dtype, np.integer) else float(img.max())
    if img.ndim == 2:
        f = img.astype(np.float64)
        s = float((img >= top).mean())
        return {k: f for k in "RGB"}, {k: s for k in "RGB"}
    planes = {k: img[..., i].astype(np.float64) for i, k in enumerate("RGB")}
    sat = {k: float((img[..., i] >= top).mean()) for i, k in enumerate("RGB")}
    return planes, sat


def load_frame(path: Path, channel: str = "Auto", rotate: int = 0) -> Frame:
    path = Path(path)
    planes, sat = (_raw_planes if path.suffix.lower() in RAW_EXT else _img_planes)(path)
    if channel == "Auto":                     # brightest channel = laser colour
        channel = max("RGB", key=lambda k: planes[k].mean())
    if channel == "Luma":
        data = 0.299 * planes["R"] + 0.587 * planes["G"] + 0.114 * planes["B"]
        s = max(sat.values())
    else:
        data, s = planes[channel], sat[channel]
    data = np.clip(data, 0, None)
    if rotate % 360:
        data = np.ascontiguousarray(np.rot90(data, k=-(rotate // 90)))
    return Frame(path, data, s, channel)


def load_scan(folder: Path, channel="Auto", rotate=0, warn=None) -> list[Frame]:
    """Load every readable frame in a scan folder. Broken files (e.g. a DNG
    that was only half-copied) are skipped and reported through `warn`."""
    frames = []
    for p in list_frames(folder):
        try:
            f = load_frame(p, channel, rotate)
        except Exception as e:  # noqa: BLE001 - report and keep going
            if warn:
                warn(f"Skipped {p.name}: {e}")
            continue
        if frames and f.data.shape != frames[0].data.shape:
            if warn:
                warn(f"Skipped {p.name}: size differs from the first frame")
            continue
        channel = f.channel                   # lock "Auto" to the first frame
        frames.append(f)
    return frames


# --------------------------------------------------------------------------
# Speckle maths
# --------------------------------------------------------------------------
def spatial_contrast(img: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Speckle contrast K = sigma/mean in a win x win window. Returns (K, local mean)."""
    m = uniform_filter(img, win)
    m2 = uniform_filter(img * img, win)
    var = np.clip(m2 - m * m, 0, None)
    return np.sqrt(var) / (m + 1e-9), m


def temporal_contrast(stack: np.ndarray) -> np.ndarray:
    return stack.std(axis=0) / (stack.mean(axis=0) + 1e-9)


def local_correlation(a: np.ndarray, b: np.ndarray, win: int) -> np.ndarray:
    """Zero-normalised cross-correlation in a sliding window. 1 = identical
    speckle pattern, ~0 = fully decorrelated (motion / surface change)."""
    ma, mb = uniform_filter(a, win), uniform_filter(b, win)
    cov = uniform_filter(a * b, win) - ma * mb
    va = np.clip(uniform_filter(a * a, win) - ma * ma, 0, None)
    vb = np.clip(uniform_filter(b * b, win) - mb * mb, 0, None)
    return np.clip(cov / (np.sqrt(va * vb) + 1e-9), -1, 1)


def find_shift(a: np.ndarray, b: np.ndarray, max_frac=0.1) -> tuple[int, int] | None:
    """Integer (dy, dx) such that b[y, x] ~ a[y - dy, x - dx], by phase correlation.
    Returns None when there is no clear peak or the shift is implausibly large
    (different scenes) - aligning would then do more harm than good."""
    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-12
    c = np.fft.irfft2(r, s=a.shape)
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    if c[dy, dx] < c.mean() + 8 * c.std():
        return None
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    if abs(dy) > max_frac * h or abs(dx) > max_frac * w:
        return None
    return int(dy), int(dx)


def crop_overlap(a: np.ndarray, b: np.ndarray, dy: int, dx: int):
    """Crop a and b to the region where they overlap after the shift."""
    h, w = a.shape
    ya, yb = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xa, xb = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    return (ya, xa), (yb, xb)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
@dataclass
class Settings:
    channel: str = "Auto"
    rotate: int = 0
    win: int = 7
    align: bool = True
    min_signal: float = 0.03      # fraction of the bright level a pixel needs to count


@dataclass
class Comparison:
    label_a: str
    label_b: str
    channel: str
    intensity_a: np.ndarray
    intensity_b: np.ndarray
    k_a: np.ndarray
    k_b: np.ndarray
    dk_rel: np.ndarray            # (K_b - K_a) / K_a
    corr: np.ndarray              # local correlation of the speckle patterns
    valid: np.ndarray             # enough light in both scans
    shift: tuple[int, int]
    sat_a: float
    sat_b: float
    n_a: int
    n_b: int
    temporal_k: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def flow_a(self):
        return 1.0 / (self.k_a ** 2 + 1e-9)

    @property
    def flow_b(self):
        return 1.0 / (self.k_b ** 2 + 1e-9)


def _stack_stats(frames: list[Frame], win: int):
    stack = np.stack([f.data for f in frames])
    ks = []
    for f in frames:
        k, _ = spatial_contrast(f.data, win)
        ks.append(k)
    return stack.mean(axis=0), np.mean(ks, axis=0), stack


def compare(frames_a: list[Frame], frames_b: list[Frame], s: Settings,
            label_a="A", label_b="B", progress=None,
            temporal_frames: list[Frame] | None = None) -> Comparison:
    if not frames_a or not frames_b:
        raise ValueError("Both sides need at least one readable frame")
    step = progress or (lambda *_: None)

    step("Speckle contrast (reference)...", 0.25)
    ia, ka, _ = _stack_stats(frames_a, s.win)
    step("Speckle contrast (compare)...", 0.45)
    ib, kb, _ = _stack_stats(frames_b, s.win)
    if ia.shape != ib.shape:
        raise ValueError(f"Image sizes differ: {ia.shape} vs {ib.shape}")

    shift = (0, 0)
    notes = []
    first_a, first_b = frames_a[0].data, frames_b[0].data
    if s.align:
        step("Aligning...", 0.6)
        found = find_shift(ia, ib)
        if found is None:
            notes.append("auto-align found no reliable match - compared unaligned")
        else:
            shift = found
        if shift != (0, 0):
            sa, sb = crop_overlap(ia, ib, *shift)
            ia, ka, first_a = ia[sa], ka[sa], first_a[sa]
            ib, kb, first_b = ib[sb], kb[sb], first_b[sb]

    step("Correlation map...", 0.75)
    corr = local_correlation(first_a, first_b, max(s.win * 2 + 1, 9))

    bright = max(np.percentile(ia, 99.5), np.percentile(ib, 99.5), 1.0)
    level = max(s.min_signal * bright, 2.0)
    ma = uniform_filter(ia, s.win)
    mb = uniform_filter(ib, s.win)
    valid = (ma > level) & (mb > level)

    dk = (kb - ka) / (ka + 1e-9)

    tk = None
    tf = temporal_frames or frames_a
    if len(tf) > 1:
        tk = temporal_contrast(np.stack([f.data for f in tf]))
        if shift != (0, 0):
            tk = tk[sa]

    step("Done", 1.0)
    return Comparison(
        label_a, label_b, frames_a[0].channel, ia, ib, ka, kb, dk, corr, valid,
        shift, max(f.saturated for f in frames_a), max(f.saturated for f in frames_b),
        len(frames_a), len(frames_b), tk, notes)


@dataclass
class Verdict:
    mask: np.ndarray
    changed_pct: float
    changed: bool
    mean_k_a: float
    mean_k_b: float
    dk_pct: float
    mean_corr: float
    valid_pct: float


def evaluate(c: Comparison, method: str, k_thr: float, corr_thr: float,
             min_area_pct: float) -> Verdict:
    """Turn the maps into a change mask + verdict. Cheap, so it can run on every
    slider move. method: 'contrast' (|dK/K| > k_thr), 'correlation'
    (corr < corr_thr) or 'combined' (either)."""
    v = c.valid
    by_k = np.abs(c.dk_rel) > k_thr
    by_c = c.corr < corr_thr
    if method == "correlation":
        raw = v & by_c
    elif method == "contrast":
        raw = v & by_k
    else:
        raw = v & (by_k | by_c)
    mask = binary_opening(raw, iterations=2)  # drop isolated noisy pixels
    nvalid = max(int(v.sum()), 1)
    pct = 100.0 * mask.sum() / nvalid
    if v.any():
        mka, mkb = float(c.k_a[v].mean()), float(c.k_b[v].mean())
        mcorr = float(c.corr[v].mean())
    else:
        mka = mkb = mcorr = float("nan")
    return Verdict(mask, pct, pct > min_area_pct, mka, mkb,
                   100.0 * (mkb - mka) / (mka + 1e-9), mcorr, 100.0 * v.mean())


def roi_stats(c: Comparison, y0, y1, x0, x1) -> dict:
    sl = (slice(y0, y1), slice(x0, x1))
    v = c.valid[sl]
    if not v.any():
        return {}
    ka, kb = c.k_a[sl][v].mean(), c.k_b[sl][v].mean()
    return {
        "K_ref": float(ka), "K_cmp": float(kb),
        "dK_pct": float(100 * (kb - ka) / (ka + 1e-9)),
        "corr": float(c.corr[sl][v].mean()),
        "flow_ratio": float((1 / kb ** 2) / (1 / ka ** 2)),
        "pixels": int(v.sum()),
    }
