#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 YokoEnhancer  -  photo & video enhancement for your laptop
==============================================================================
 Natural, photorealistic enhancement up to 4K / 8K.  No plastic skin, no
 oil-paint look.  Everything (UI included) is drawn by this single file.

 INSTALL (once)  - Python 3.10 / 3.11 / 3.12 on Windows 10/11
 ---------------------------------------------------------------------------
   pip install PySide6 opencv-python numpy spandrel pillow
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

 (the second line installs the NVIDIA GPU build of PyTorch - needed for your
  RTX 3050.  A CPU-only torch will work but is very slow.)

 VIDEO needs FFmpeg:      winget install Gyan.FFmpeg      (or: pip install imageio-ffmpeg)

 RUN
 ---------------------------------------------------------------------------
   python YokoEnhancer.py

 The enhancement files (about 0.1 - 0.2 GB each) are downloaded automatically
 the first time you use a quality level.  They are stored in
   C:\\Users\\<you>\\YokoEnhancer\\models
 You can also copy the .pth files there by hand (offline use).
==============================================================================
"""
from __future__ import annotations

import os
import sys
import re
import json
import math
import time
import shutil
import weakref
import queue
import logging
import threading
import subprocess
import traceback
import tempfile
import platform
import urllib.request
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict

APP_NAME = "YokoEnhancer"
APP_VERSION = "1.0"

# ----------------------------------------------------------------------------
#  Dependency check (friendly message instead of a traceback)
# ----------------------------------------------------------------------------
_MISSING: List[str] = []
try:
    import numpy as np
except Exception:
    _MISSING.append("numpy")
try:
    import cv2
except Exception:
    _MISSING.append("opencv-python")
try:
    import torch
    import torch.nn.functional as F
except Exception:
    _MISSING.append("torch  (pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124)")
try:
    import spandrel  # noqa: F401
except Exception:
    _MISSING.append("spandrel")
try:
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
except Exception:
    _MISSING.append("PySide6")


def _fatal_missing(missing: List[str]) -> None:
    msg = (
        f"{APP_NAME} needs a few things installed first:\n\n  - "
        + "\n  - ".join(missing)
        + "\n\nOpen a terminal and run:\n  pip install PySide6 opencv-python numpy spandrel pillow\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
    )
    print(msg)
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(APP_NAME, msg)
        root.destroy()
    except Exception:
        pass
    sys.exit(1)


if _MISSING:
    _fatal_missing(_MISSING)

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import (Qt, QRectF, QPointF, QSize, QTimer, Signal, QThread, QObject,
                            QEasingCurve, QVariantAnimation, QRect, QPoint, QUrl)
from PySide6.QtGui import (QPainter, QColor, QLinearGradient, QRadialGradient, QPen, QBrush,
                           QPainterPath, QFont, QImage, QPixmap, QIcon, QCursor,
                           QConicalGradient, QFontMetrics, QDesktopServices)
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QFileDialog, QListWidget, QListWidgetItem, QStyledItemDelegate,
                               QDialog, QScrollArea, QFrame, QMenu, QAbstractButton, QSizePolicy,
                               QAbstractItemView)

# ----------------------------------------------------------------------------
#  Paths, logging
# ----------------------------------------------------------------------------
def app_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


DATA_DIR = Path(os.environ.get("YOKO_HOME", str(Path.home() / "YokoEnhancer")))
MODELS_DIR = DATA_DIR / "models"
WORK_DIR = DATA_DIR / "work"            # enhanced results wait here until you press Download
SETTINGS_FILE = DATA_DIR / "settings.json"
LOG_FILE = DATA_DIR / "yoko.log"
for _d in (DATA_DIR, MODELS_DIR, WORK_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

try:                                    # leftovers from an earlier session
    for _f in WORK_DIR.iterdir():
        try:
            if _f.is_file():
                _f.unlink()
        except Exception:
            pass
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
try:
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)
except Exception:
    pass
log = logging.getLogger(APP_NAME)

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg"}


class Cancelled(Exception):
    """Raised inside worker threads when the user presses Stop."""


# ----------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    quality: str = "maximum"        # maximum | balanced | fast | turbo (video only)
    photo_quality: str = "maximum"  # last choice among the three photo levels
    target: str = "4k"              # 4k | 8k | 2x | 4x | keep
    remove_blur: bool = True
    blur_strength: int = 45
    remove_grain: bool = False
    grain_strength: int = 35
    detail: int = 35
    texture: int = 40
    output_format: str = "png"      # png | jpg | webp | tiff
    jpg_quality: int = 96
    output_dir: str = ""            # download folder; empty -> your Downloads folder
    use_gpu: bool = True
    video_quality: str = "high"     # high | standard | small
    video_codec: str = "h264"       # h264 | h265
    steady_video: bool = True
    video_format: str = "mp4"      # mp4 | mov | mkv
    theme: str = "ivory"
    ambient: bool = True            # animated background objects
    gpu_index: int = -1             # -1 = pick the strongest graphics card automatically
    last_dir: str = ""

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            if SETTINGS_FILE.exists():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                names = {f.name for f in fields(cls)}
                for k, v in data.items():
                    if k in names:
                        setattr(s, k, v)
        except Exception as e:
            log.warning("Could not read settings: %s", e)
        return s

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Could not save settings: %s", e)


# ----------------------------------------------------------------------------
#  Model catalogue  (the UI only ever says Maximum / Balanced / Fast)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    files: Tuple[str, ...]
    urls: Tuple[str, ...]
    heavy: int          # 1 light ... 3 very heavy (drives tile size)


_GH = "https://github.com/Phhofm/models/releases/download"
MODELS: Dict[str, ModelSpec] = {
    "maximum": ModelSpec(
        "maximum", "Maximum",
        ("4xNomos8kDAT.pth", "4xNomos8kDAT.safetensors"),
        (f"{_GH}/4xNomos8kDAT/4xNomos8kDAT.pth",),
        3,
    ),
    "balanced": ModelSpec(
        "balanced", "Balanced",
        ("4xNomos8kSCHAT-S.pth", "4xNomos8kSCHAT-S.safetensors"),
        (f"{_GH}/4xNomos8kSCHAT/4xNomos8kSCHAT-S.pth",),
        2,
    ),
    "fast": ModelSpec(
        "fast", "Fast",
        ("4xNomosWebPhoto_RealPLKSR.pth", "4xNomosWebPhoto_RealPLKSR.safetensors"),
        (f"{_GH}/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth",),
        1,
    ),
}


def find_model_file(spec: ModelSpec) -> Optional[Path]:
    for folder in (MODELS_DIR, app_dir() / "models", app_dir()):
        for name in spec.files:
            p = folder / name
            try:
                if p.exists() and p.stat().st_size > 1_000_000:
                    return p
            except Exception:
                pass
    return None


def download_file(urls, dest: Path, progress_cb: Optional[Callable[[int, int], None]] = None,
                  cancel: Optional[threading.Event] = None) -> None:
    last_err: Optional[Exception] = None
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"})
            with urllib.request.urlopen(req, timeout=40) as r:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        chunk = r.read(1 << 17)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb:
                            progress_cb(done, total)
            if tmp.stat().st_size < 1_000_000:
                raise IOError("Downloaded file is too small")
            tmp.replace(dest)
            return
        except Cancelled:
            try:
                tmp.unlink()
            except Exception:
                pass
            raise
        except Exception as e:  # try next mirror
            last_err = e
            log.warning("Download failed from %s: %s", url, e)
            try:
                tmp.unlink()
            except Exception:
                pass
    raise RuntimeError(f"Download failed: {last_err}")


# ----------------------------------------------------------------------------
#  Small utilities
# ----------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def open_in_folder(path: str) -> None:
    try:
        p = Path(path)
        if sys.platform.startswith("win"):
            if p.exists():
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                os.startfile(str(p.parent))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent)])
    except Exception as e:
        log.warning("open_in_folder: %s", e)


def open_folder(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def default_download_dir() -> Path:
    d = Path.home() / "Downloads"
    return d if d.exists() else Path.home()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 2
    while True:
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ============================================================================
#  IMAGE INPUT / OUTPUT
# ============================================================================
def _apply_orientation(img: "np.ndarray", orient: int) -> "np.ndarray":
    if orient == 2:
        return cv2.flip(img, 1)
    if orient == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orient == 4:
        return cv2.flip(img, 0)
    if orient == 5:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
    if orient == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orient == 7:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orient == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _exif_orientation(path: str) -> int:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        return 1


def _to_uint8(img: "np.ndarray") -> "np.ndarray":
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    return np.clip(img.astype(np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def imread_any(path) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    """Read any common image. Returns (BGR uint8, alpha uint8 or None)."""
    p = str(path)
    img = None
    try:
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
            # IMREAD_COLOR also applies the EXIF rotation of phone photos
            img2 = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img2 is not None:
                img = img2
        elif img is not None and img.ndim == 2 and img.dtype == np.uint8:
            img2 = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img2 is not None:
                img = img2
        elif img is not None:
            img = _apply_orientation(img, _exif_orientation(p))
    except Exception as e:
        log.warning("cv2 decode failed for %s: %s", p, e)
        img = None

    if img is None:
        try:
            from PIL import Image, ImageOps
            im = ImageOps.exif_transpose(Image.open(p))
            has_alpha = "A" in im.getbands()
            im = im.convert("RGBA" if has_alpha else "RGB")
            arr = np.array(im)
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if has_alpha else cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise IOError(f"Cannot open this picture ({e})")

    img = _to_uint8(img)
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 2:
        g = img[:, :, 0]
        alpha = img[:, :, 1]
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3].copy()
        img = img[:, :, :3].copy()
    if alpha is not None and int(alpha.min()) == 255:
        alpha = None
    return np.ascontiguousarray(img), alpha


def imwrite_any(path: Path, bgr: "np.ndarray", alpha: Optional["np.ndarray"], fmt: str,
                jpg_quality: int = 96) -> None:
    fmt = fmt.lower()
    ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp", "tiff": ".tif"}.get(fmt, ".png")
    params: List[int] = []
    out = bgr
    if fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
    elif fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
        sampling = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        if sampling is not None:
            params += [sampling, 0x111111]       # 4:4:4 chroma - keeps colour detail
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 100]
    elif fmt == "tiff":
        params = []
    if alpha is not None and fmt in ("png", "webp", "tiff"):
        out = np.dstack([bgr, alpha])
    ok, buf = cv2.imencode(ext, out, params)
    if not ok:
        raise IOError("Could not encode the output picture")
    buf.tofile(str(path))


def bgr_to_qimage(arr: "np.ndarray") -> QImage:
    h, w = arr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def make_thumbnail(path: str, kind: str, size: int = 220) -> Optional["np.ndarray"]:
    """Small BGR thumbnail (cover-cropped square)."""
    try:
        if kind == "video":
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n > 20:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.1))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            img = frame
        else:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
            if img is None or min(img.shape[:2]) < size // 2:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                img, _ = imread_any(path)
        h, w = img.shape[:2]
        s = size / min(h, w)
        img = cv2.resize(img, (max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y0 = (h - size) // 2 if h > size else 0
        x0 = (w - size) // 2 if w > size else 0
        return np.ascontiguousarray(img[y0:y0 + size, x0:x0 + size])
    except Exception as e:
        log.warning("thumbnail failed for %s: %s", path, e)
        return None


# ============================================================================
#  OUTPUT SIZE PLANNING
# ============================================================================
MAX_OUTPUT_PIXELS = 36_000_000


def plan_output(w: int, h: int, target: str, even: bool = False) -> Tuple[int, int, float]:
    """Return (out_w, out_h, scale) for a chosen target."""
    long_side = max(w, h)
    if target == "4k":
        s = max(1.0, 3840.0 / long_side)
    elif target == "8k":
        s = max(1.0, 7680.0 / long_side)
    elif target == "2x":
        s = 2.0
    elif target == "4x":
        s = 4.0
    else:
        s = 1.0
    if w * h * s * s > MAX_OUTPUT_PIXELS:
        s = math.sqrt(MAX_OUTPUT_PIXELS / float(w * h))
    ow = max(2, int(round(w * s)))
    oh = max(2, int(round(h * s)))
    if even:
        ow -= ow % 2
        oh -= oh % 2
    return ow, oh, s


def target_label(target: str) -> str:
    return {"4k": "4K", "8k": "8K", "2x": "2x", "4x": "4x", "keep": "Clean"}.get(target, "")


# ============================================================================
#  ENHANCEMENT PARAMETERS + POST PROCESSING  (the "photoreal" part)
# ============================================================================
@dataclass
class EnhanceParams:
    quality: str = "maximum"
    target: str = "4k"
    remove_blur: bool = True
    blur_strength: int = 45
    remove_grain: bool = False
    grain_strength: int = 35
    detail: int = 35
    texture: int = 25

    @classmethod
    def from_settings(cls, s: Settings) -> "EnhanceParams":
        return cls(s.quality, s.target, s.remove_blur, s.blur_strength, s.remove_grain,
                   s.grain_strength, s.detail, s.texture)


def _res_scale(out_w: int, out_h: int) -> float:
    """Filter sizes are expressed for a 4K frame and scaled for other sizes."""
    return float(clamp(max(out_w, out_h) / 3840.0, 0.5, 2.5))


def denoise_grain(bgr: "np.ndarray", strength: int) -> "np.ndarray":
    """Remove unwanted grain / sensor noise BEFORE enlarging (so it is never
    enlarged into blotches).  Luma is treated gently, colour noise strongly."""
    s = clamp(strength, 0, 100) / 100.0
    if s <= 0:
        return bgr
    h_luma = 1.5 + 8.5 * s
    h_color = 2.0 + 11.0 * s
    h, w = bgr.shape[:2]
    if h * w > 16_000_000:
        # very large input: process in horizontal bands to keep memory sane
        out = np.empty_like(bgr)
        band, pad = 1024, 16
        for y in range(0, h, band):
            y0, y1 = max(0, y - pad), min(h, y + band + pad)
            piece = cv2.fastNlMeansDenoisingColored(bgr[y0:y1], None, h_luma, h_color, 5, 15)
            out[y:min(h, y + band)] = piece[y - y0:y - y0 + min(band, h - y)]
        den = out
    else:
        den = cv2.fastNlMeansDenoisingColored(bgr, None, h_luma, h_color, 5, 15)
    # keep a little of the original micro structure so skin does not go waxy
    keep = 0.18 * (1.0 - s)
    if keep > 0.01:
        den = cv2.addWeighted(den, 1.0 - keep, bgr, keep, 0)
    return den


# ---------------------------------------------------------------------------
#  GPU image operators.  All of these work on float32 tensors shaped (N,C,H,W)
#  and stay on the graphics card, so 4K frames cost milliseconds, not seconds.
# ---------------------------------------------------------------------------
_KCACHE: Dict[Tuple[float, str], Tuple["torch.Tensor", "torch.Tensor", int]] = {}
_SOBEL: Dict[str, Tuple["torch.Tensor", "torch.Tensor"]] = {}


def _tsmooth(x: "torch.Tensor") -> "torch.Tensor":
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _luma(x: "torch.Tensor") -> "torch.Tensor":
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def _gauss_kernels(sigma: float, device) -> Tuple["torch.Tensor", "torch.Tensor", int]:
    key = (round(float(sigma), 3), str(device))
    hit = _KCACHE.get(key)
    if hit is not None:
        return hit
    sigma = max(0.2, float(sigma))
    r = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    val = (k.view(1, 1, 1, -1).contiguous(), k.view(1, 1, -1, 1).contiguous(), r)
    _KCACHE[key] = val
    return val


def gblur(x: "torch.Tensor", sigma: float) -> "torch.Tensor":
    """Separable gaussian blur for (N,1,H,W)."""
    kh, kv, r = _gauss_kernels(sigma, x.device)
    mode = "reflect" if r < min(x.shape[-2:]) else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), kh)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), kv)
    return x


def sobel_mag(x: "torch.Tensor") -> "torch.Tensor":
    key = str(x.device)
    if key not in _SOBEL:
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                          device=x.device).view(1, 1, 3, 3)
        _SOBEL[key] = (kx, kx.transpose(2, 3).contiguous())
    kx, ky = _SOBEL[key]
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(p, kx)
    gy = F.conv2d(p, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _dilate(x: "torch.Tensor", k: int) -> "torch.Tensor":
    p = k // 2
    return F.max_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), k, 1)


def _erode(x: "torch.Tensor", k: int) -> "torch.Tensor":
    return -_dilate(-x, k)


def deblur_y(y: "torch.Tensor", strength: int, sc: float, fast: bool = False) -> "torch.Tensor":
    """Edge-aware deblurring of luminance: Richardson-Lucy, applied only on real
    edges, with a local min/max guard so there are no halos."""
    a = clamp(strength, 0, 100) / 100.0
    if a <= 0:
        return y
    yc = y * 0.96 + 0.02
    sigma = (0.55 + 0.95 * a) * sc
    iters = 5 + int(round(7 * a))
    if fast:
        iters = min(iters, 6)
    est = yc.clone()
    for _ in range(iters):
        est = est * gblur(yc / (gblur(est, sigma) + 1e-4), sigma)
    k = max(3, int(round(2.0 * sc)) * 2 + 1)
    lo, hi = _erode(y, k), _dilate(y, k)
    diff = (est - yc).clamp(-0.25, 0.25)
    cand = torch.minimum(torch.maximum(y + diff, lo - 0.015), hi + 0.015)
    diff = cand - y
    grad = sobel_mag(gblur(y, 1.0 * sc))
    mask = gblur(_tsmooth((grad - 0.006) / 0.05), 1.5 * sc)
    return (y + diff * mask * (0.35 + 0.65 * a)).clamp(0.0, 1.0)


def detail_y(y: "torch.Tensor", amount: int, sc: float) -> "torch.Tensor":
    a = clamp(amount, 0, 100) / 100.0
    if a <= 0:
        return y
    b1 = gblur(y, 0.9 * sc)
    b2 = gblur(y, 2.6 * sc)
    lim = 0.05
    fine = lim * torch.tanh((y - b1) / lim)
    mid = lim * torch.tanh((b1 - b2) / lim)
    gate = _tsmooth((gblur(fine.abs(), 2.0 * sc) - 0.0015) / 0.012)
    return (y + a * (0.85 * fine + 0.50 * mid) * (0.35 + 0.65 * gate)).clamp(0.0, 1.0)


def estimate_noise(ys: "torch.Tensor") -> float:
    """How noisy / grainy is the ORIGINAL photo (0.001 .. 0.03).  The same
    amount of natural grain is later put back so the result matches the
    photo's own character instead of looking airbrushed."""
    d = ys - gblur(ys, 1.0)
    gm = sobel_mag(gblur(ys, 1.2))
    thr = gm.flatten()[::7].median()
    vals = d[gm <= thr].abs()
    if vals.numel() < 100:
        return 0.005
    sig = 1.4826 * float(vals.median()) * 1.25
    return float(clamp(sig, 0.001, 0.03))


def texture_y(y: "torch.Tensor", ys: "torch.Tensor", amount: int, sc: float, f_up: float,
              sigma_n: float, grain_keep: float, gen) -> "torch.Tensor":
    """Anti-plastic step.
       1. real texture from the original photo is returned to areas the AI left
          suspiciously smooth (skin, walls, sky);
       2. photographic grain, with the same strength as the original photo and
          the right grain size for the enlargement, is added, weighted to mid-tones."""
    t = clamp(amount, 0, 100) / 100.0
    if t <= 0:
        return y
    band = gblur(ys, 1.1 * sc) - gblur(ys, 3.2 * sc)
    mean = gblur(y, 2.2 * sc)
    var = (gblur(y * y, 2.2 * sc) - mean * mean).clamp_min(0.0)
    flat = 1.0 - _tsmooth((torch.sqrt(var) - 0.004) / 0.03)
    y = y + band * flat * (0.95 * t)
    n = torch.randn(y.shape, device=y.device, dtype=torch.float32, generator=gen)
    grain_sigma = clamp(0.5 * (max(f_up, 1.0) ** 0.55), 0.6, 2.4) * sc
    n = gblur(n, grain_sigma)
    n = n / (n.std() + 1e-6)
    amp = t * clamp(sigma_n * 0.9 + 0.004, 0.003, 0.02) * grain_keep
    lum_w = 0.30 + 0.70 * 4.0 * y * (1.0 - y)
    return (y + n * lum_w * amp).clamp(0.0, 1.0)


def gpu_denoise(rgb: "torch.Tensor", strength: int) -> "torch.Tensor":
    """Fast edge-preserving noise removal on the graphics card (used for video)."""
    s = clamp(strength, 0, 100) / 100.0
    if s <= 0:
        return rgb
    sig = 0.02 + 0.09 * s
    H, W = rgb.shape[-2:]
    p = F.pad(rgb, (2, 2, 2, 2), mode="replicate")
    acc = torch.zeros_like(rgb)
    wsum = torch.zeros_like(rgb[:, :1])
    inv = 1.0 / (2.0 * sig * sig * 3.0)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            nb = p[:, :, 2 + dy:2 + dy + H, 2 + dx:2 + dx + W]
            d = ((nb - rgb) ** 2).sum(1, keepdim=True)
            w = torch.exp(-d * inv) * math.exp(-(dx * dx + dy * dy) / 4.5)
            acc += nb * w
            wsum += w
    den = acc / wsum
    keep = 0.18 * (1.0 - s)
    return den * (1.0 - keep) + rgb * keep


class TemporalGPU:
    """Calms flicker between video frames in still areas.  Runs on the GPU."""

    def __init__(self) -> None:
        self.pg: Optional["torch.Tensor"] = None
        self.po: Optional["torch.Tensor"] = None

    def apply(self, gray: "torch.Tensor", res: "torch.Tensor") -> "torch.Tensor":
        if self.pg is None or self.po is None or self.pg.shape != gray.shape or self.po.shape != res.shape:
            self.pg, self.po = gray, res
            return res
        diff = (gray - self.pg).abs()
        if float(diff.mean()) > 22.0 / 255.0:                 # scene cut
            self.pg, self.po = gray, res
            return res
        wm = torch.exp(-torch.square(diff / (7.0 / 255.0))) * 0.45
        wm = gblur(wm, 1.5)
        wm = F.interpolate(wm, size=res.shape[-2:], mode="bilinear", align_corners=False)
        out = (res.float() * (1.0 - wm) + self.po.float() * wm).to(res.dtype)
        self.pg, self.po = gray, out
        return out


# ============================================================================
#  ENGINE  (tiled, memory-safe inference for 4 GB graphics cards)
# ============================================================================
def _is_oom(e: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


class Engine:
    """Loads one enhancement model at a time and runs it tile by tile.
    * auto tile size for the graphics card, automatic fallback on out-of-memory
    * half precision with automatic fall back to full precision
    * seamless feathered blending between tiles
    * tiles are resized straight to the final grid, so even huge images stay
      within RAM."""

    def __init__(self, use_gpu: bool = True, gpu_index: int = -1):
        self.desc = None
        self.set_device(use_gpu, gpu_index)
        self.key: Optional[str] = None
        self.scale = 4
        self.fp16 = False
        self.max_side = 10 ** 6
        self.heavy = 1
        self.req = None
        self.video_mode = False
        self.check_finite = True

    # ------------------------------------------------------------------ device
    @staticmethod
    def gpu_list() -> List[Tuple[int, str, float]]:
        """Every usable graphics card in this PC: (index, name, VRAM in GB)."""
        out: List[Tuple[int, str, float]] = []
        try:
            if torch.cuda.is_available():
                for k in range(torch.cuda.device_count()):
                    pr = torch.cuda.get_device_properties(k)
                    nm = pr.name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                    out.append((k, nm, pr.total_memory / 2 ** 30))
        except Exception:
            pass
        return out

    def set_device(self, use_gpu: bool, index: int = -1) -> None:
        gpus = self.gpu_list()
        if use_gpu and gpus:
            ids = [g[0] for g in gpus]
            idx = index if index in ids else max(gpus, key=lambda g: g[2])[0]
            self.device = torch.device("cuda", idx)
            try:
                torch.cuda.set_device(idx)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
        else:
            self.device = torch.device("cpu")
        if self.desc is not None:
            self.desc = None
            self.key = None
            self.free_memory()

    @property
    def on_gpu(self) -> bool:
        return self.device.type == "cuda"

    def _idx(self) -> int:
        return self.device.index if self.device.index is not None else 0

    def gpu_name(self) -> str:
        if self.on_gpu:
            try:
                nm = torch.cuda.get_device_properties(self._idx()).name
                return nm.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            except Exception:
                return "Graphics card"
        return "Processor"

    def vram_total_gb(self) -> float:
        if not self.on_gpu:
            return 0.0
        try:
            return torch.cuda.get_device_properties(self._idx()).total_memory / 2 ** 30
        except Exception:
            return 0.0

    def vram_free_gb(self) -> float:
        if not self.on_gpu:
            return 0.0
        try:
            return torch.cuda.mem_get_info(self._idx())[0] / 2 ** 30
        except Exception:
            return 2.0

    def free_memory(self) -> None:
        try:
            if torch.cuda.is_available():
                if self.on_gpu:
                    with torch.cuda.device(self._idx()):
                        torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------- model
    def ensure_model(self, spec: ModelSpec, status: Callable[[str, float], None],
                     cancel: Optional[threading.Event]) -> Path:
        path = find_model_file(spec)
        if path is not None:
            return path
        dest = MODELS_DIR / spec.files[0]

        def prog(done: int, total: int) -> None:
            frac = (done / total) if total else 0.0
            status(f"Setting up for first use  {int(frac * 100)}%", frac)

        status("Setting up for first use", 0.0)
        download_file(spec.urls, dest, prog, cancel)
        return dest

    def load(self, spec: ModelSpec, status: Callable[[str, float], None],
             cancel: Optional[threading.Event]) -> None:
        if self.key == spec.key and self.desc is not None:
            return
        path = self.ensure_model(spec, status, cancel)
        status("Preparing", 0.0)
        from spandrel import ModelLoader, ImageModelDescriptor
        self.desc = None
        self.free_memory()
        desc = ModelLoader().load_from_file(str(path))
        if not isinstance(desc, ImageModelDescriptor):
            raise RuntimeError("This enhancement file is not usable")
        desc.to(self.device)
        desc.eval()
        self.fp16 = False
        if self.on_gpu and getattr(desc, "supports_half", False):
            try:
                desc.half()
                self.fp16 = True
            except Exception:
                self.fp16 = False
        self.desc = desc
        self.key = spec.key
        self.heavy = spec.heavy
        self.scale = int(getattr(desc, "scale", 4) or 4)
        self.req = getattr(desc, "size_requirements", None)
        self.max_side = 10 ** 6
        log.info("Loaded %s on %s (fp16=%s, scale=%d)", path.name, self.device, self.fp16, self.scale)

    # --------------------------------------------------------------- tile size
    def auto_tile(self, video: bool = False) -> int:
        if not self.on_gpu:
            return 128 if self.heavy >= 3 else 192
        free = self.vram_free_gb()
        h = self.heavy
        table = {
            3: [(7.0, 320), (4.5, 256), (3.4, 192), (2.5, 160), (1.7, 128)],
            2: [(7.0, 512), (4.5, 384), (3.2, 288), (2.2, 224), (1.5, 160)],
            1: [(6.0, 640), (4.0, 512), (3.0, 384), (2.0, 288), (1.3, 192)],
        }[max(1, min(3, h))]
        if video and h == 1:
            table = [(5.0, 768), (3.0, 512), (2.2, 384), (1.5, 256), (1.0, 192)]
        for need, tile in table:
            if free >= need:
                return tile
        return 96

    def _pad_amounts(self, h: int, w: int) -> Tuple[int, int]:
        req = self.req
        m = int(getattr(req, "multiple_of", 1) or 1)
        mn = int(getattr(req, "minimum", 0) or 0)
        th, tw = max(h, mn), max(w, mn)
        th += (-th) % m
        tw += (-tw) % m
        if getattr(req, "square", False):
            t = max(th, tw)
            th = tw = t
        return th - h, tw - w

    def set_video_mode(self, on: bool) -> None:
        self.video_mode = on
        if self.on_gpu:
            try:
                torch.backends.cudnn.benchmark = bool(on)     # fixed frame size -> pick fastest kernels
            except Exception:
                pass

    # --------------------------------------------------------------- inference
    def _run(self, piece: "torch.Tensor") -> "torch.Tensor":
        """piece (1,3,h,w) float32 RGB on device  ->  (1,3,h*s,w*s) clamped 0..1"""
        h, w = piece.shape[-2:]
        s = self.scale
        t = piece.to(torch.float16 if self.fp16 else torch.float32)
        ph, pw = self._pad_amounts(h, w)
        if ph or pw:
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            t = F.pad(t, (0, pw, 0, ph), mode=mode)
        out = self.desc(t)
        return out[:, :, :h * s, :w * s].clamp(0.0, 1.0)

    def _infer(self, piece: "torch.Tensor") -> "torch.Tensor":
        h, w = piece.shape[-2:]
        if max(h, w) > self.max_side and max(h, w) > 64:
            return self._split(piece)
        try:
            res = self._run(piece)
            if self.fp16 and self.check_finite and not bool(torch.isfinite(res).all()):
                log.warning("Half precision gave invalid values - switching to full precision")
                self.desc.float()
                self.fp16 = False
                res = self._run(piece)
            return res
        except Exception as e:
            if _is_oom(e):
                self.free_memory()
                if max(h, w) <= 64:
                    raise RuntimeError("Not enough graphics memory. Close other apps and try again.")
                self.max_side = max(64, max(h, w) // 2)
                log.warning("Out of memory - tile reduced to %d", self.max_side)
                return self._split(piece)
            raise

    def _split(self, piece: "torch.Tensor") -> "torch.Tensor":
        h, w = piece.shape[-2:]
        s = self.scale
        ov = 16
        axis = 3 if w >= h else 2
        n = piece.shape[axis]
        mid = n // 2
        m0, m1 = max(1, mid - ov), min(n - 1, mid + ov)
        ra = self._infer(piece.narrow(axis, 0, m1)).float()
        rb = self._infer(piece.narrow(axis, m0, n - m0)).float()
        ow = m1 - m0
        t = (torch.arange(ow * s, device=piece.device, dtype=torch.float32) + 0.5) / (ow * s)
        t = t * t * (3.0 - 2.0 * t)
        shape = [1, 1, 1, 1]
        shape[axis] = -1
        t = t.view(shape)
        left = ra.narrow(axis, 0, m0 * s)
        blend = ra.narrow(axis, m0 * s, ow * s) * (1.0 - t) + rb.narrow(axis, 0, ow * s) * t
        right = rb.narrow(axis, ow * s, rb.shape[axis] - ow * s)
        return torch.cat([left, blend, right], dim=axis)

    @staticmethod
    def _starts(n: int, tile: int, stride: int) -> List[int]:
        if n <= tile:
            return [0]
        starts = list(range(0, n - tile, stride))
        starts.append(n - tile)
        return sorted(set(starts))

    @staticmethod
    def _ramp(n: int, device) -> "torch.Tensor":
        x = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
        return x * x * (3.0 - 2.0 * x)

    def upscale(self, src: "torch.Tensor", out_w: int, out_h: int,
                progress: Optional[Callable[[float], None]] = None,
                cancel: Optional[threading.Event] = None) -> "torch.Tensor":
        """src (1,3,H,W) float32 RGB on device -> (1,3,out_h,out_w) RGB 0..1.
        Everything stays on the graphics card."""
        if self.desc is None:
            raise RuntimeError("Engine not loaded")
        dev = src.device
        H, W = src.shape[-2:]
        fx, fy = out_w / float(W), out_h / float(H)
        overlap = 8 if (self.video_mode and self.heavy == 1) else (16 if self.heavy >= 2 else 12)
        tile = min(self.auto_tile(self.video_mode), max(H, W))
        tile = max(tile, overlap * 4)
        stride = tile - overlap
        xs = self._starts(W, tile, stride)
        ys = self._starts(H, tile, stride)
        total = len(xs) * len(ys)

        adt = torch.float16 if self.on_gpu else torch.float32
        acc = torch.zeros((1, 3, out_h, out_w), dtype=adt, device=dev)
        wacc = torch.zeros((1, 1, out_h, out_w), dtype=adt, device=dev)
        done = 0
        for y0 in ys:
            y1 = min(H, y0 + tile)
            Y0 = int(round(y0 * fy))
            Y1 = int(round(y1 * fy)) if y1 < H else out_h
            for x0 in xs:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                x1 = min(W, x0 + tile)
                X0 = int(round(x0 * fx))
                X1 = int(round(x1 * fx)) if x1 < W else out_w
                res = self._infer(src[:, :, y0:y1, x0:x1])
                tw, th = X1 - X0, Y1 - Y0
                if res.shape[-1] != tw or res.shape[-2] != th:
                    res = F.interpolate(res.float(), size=(th, tw), mode="bicubic",
                                        align_corners=False, antialias=True).clamp_(0.0, 1.0)
                wx = torch.ones(tw, device=dev)
                wy = torch.ones(th, device=dev)
                ovx = max(1, min(int(round(overlap * fx)), tw // 2))
                ovy = max(1, min(int(round(overlap * fy)), th // 2))
                if x0 > 0:
                    wx[:ovx] = torch.minimum(wx[:ovx], self._ramp(ovx, dev))
                if x1 < W:
                    wx[-ovx:] = torch.minimum(wx[-ovx:], self._ramp(ovx, dev).flip(0))
                if y0 > 0:
                    wy[:ovy] = torch.minimum(wy[:ovy], self._ramp(ovy, dev))
                if y1 < H:
                    wy[-ovy:] = torch.minimum(wy[-ovy:], self._ramp(ovy, dev).flip(0))
                wgt = (wy[:, None] * wx[None, :]).view(1, 1, th, tw)
                acc[:, :, Y0:Y1, X0:X1] += (res.float() * wgt).to(adt)
                wacc[:, :, Y0:Y1, X0:X1] += wgt.to(adt)
                done += 1
                if progress:
                    progress(done / total)
        acc.div_(wacc.clamp_min(1e-3))
        return acc


# ============================================================================
#  FULL PIPELINE FOR ONE FRAME / PHOTO  (runs on the graphics card)
# ============================================================================
def _finish(up: "torch.Tensor", src: "torch.Tensor", params: EnhanceParams, sc: float,
            seed: int, noise_cache: Optional[List[float]] = None, video: bool = False) -> "torch.Tensor":
    """Deblur / detail / natural texture on the luminance, band by band."""
    if not (params.remove_blur or params.detail > 0 or params.texture > 0):
        return up
    dev = up.device
    H, W = up.shape[-2:]
    ysrc = _luma(src)
    ys_full = None
    sigma_n = 0.005
    if params.texture > 0:
        if noise_cache and (noise_cache[1] % 15) != 0:
            sigma_n = noise_cache[0]
            noise_cache[1] += 1
        else:
            sigma_n = estimate_noise(ysrc)
            if noise_cache is not None:
                noise_cache[:] = [sigma_n, 1]
        ys_full = F.interpolate(ysrc, size=(H, W), mode="bicubic", align_corners=False)
    f_up = W / float(src.shape[-1])
    grain_keep = 1.0 - (0.75 * params.grain_strength / 100.0 if params.remove_grain else 0.0)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234567 + int(seed))
    halo = int(clamp(48 * sc, 24, 128))
    rows = max(128, int(9e6 / W) - 2 * halo)
    res = torch.empty_like(up)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        a, b = max(0, r0 - halo), min(H, r1 + halo)
        band = up[:, :, a:b, :].float()
        y0 = _luma(band)
        y = y0
        if params.remove_blur:
            y = deblur_y(y, params.blur_strength, sc, video)
        if params.detail > 0:
            y = detail_y(y, params.detail, sc)
        if params.texture > 0:
            y = texture_y(y, ys_full[:, :, a:b, :], params.texture, sc, f_up, sigma_n, grain_keep, gen)
        o = r0 - a
        n = r1 - r0
        delta = (y - y0)[:, :, o:o + n, :]
        res[:, :, r0:r1, :] = (band[:, :, o:o + n, :] + delta).clamp_(0.0, 1.0).to(up.dtype)
    return res


@torch.inference_mode()
def enhance_frame(engine: Engine, bgr: "np.ndarray", params: EnhanceParams,
                  out_w: int, out_h: int,
                  progress: Optional[Callable[[float], None]] = None,
                  cancel: Optional[threading.Event] = None,
                  seed: int = 0, stage: Optional[Callable[[str], None]] = None,
                  temporal: Optional[TemporalGPU] = None, video: bool = False,
                  noise_cache: Optional[List[float]] = None) -> "np.ndarray":
    """BGR uint8 in  ->  BGR uint8 out at (out_w, out_h)."""
    dev = engine.device
    src_np = bgr
    if params.remove_grain and not video:
        if stage:
            stage("Cleaning")
        src_np = denoise_grain(bgr, params.grain_strength)      # best quality for photos
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    src = torch.from_numpy(np.ascontiguousarray(src_np[:, :, ::-1])).to(dev)
    src = src.permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255.0)
    if params.remove_grain and video:
        src = gpu_denoise(src, params.grain_strength)           # fast version for video

    if stage:
        stage("Enhancing")
    up = engine.upscale(src, out_w, out_h, progress, cancel)
    sc = _res_scale(out_w, out_h)

    if stage:
        stage("Finishing")
    res = _finish(up, src, params, sc, seed, noise_cache, video)
    del up
    if temporal is not None:
        res = temporal.apply(gblur(_luma(src), 1.2), res)
    out = (res.float() * 255.0 + 0.5).clamp_(0.0, 255.0).to(torch.uint8)
    out = out[0].permute(1, 2, 0).flip(-1).contiguous().cpu().numpy()
    del res, src
    return out


# ============================================================================
#  VIDEO
# ============================================================================
_ENCODER_CACHE: Dict[str, bool] = {}
_FFMPEG_CACHE: Dict[str, Optional[str]] = {}


def _popen_kwargs() -> dict:
    kw: dict = {}
    if sys.platform.startswith("win"):
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return kw


def find_ffmpeg() -> Optional[str]:
    if "path" in _FFMPEG_CACHE:
        return _FFMPEG_CACHE["path"]
    path = shutil.which("ffmpeg")
    if not path:
        for cand in (app_dir() / "ffmpeg.exe", app_dir() / "ffmpeg" / "ffmpeg.exe",
                     DATA_DIR / "ffmpeg.exe", DATA_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"):
            if cand.exists():
                path = str(cand)
                break
    if not path:
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            path = None
    _FFMPEG_CACHE["path"] = path
    return path


def _encoder_works(ffmpeg: str, name: str, w: int, h: int) -> bool:
    key = f"{name}:{w}x{h}"
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    ok = False
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             f"color=black:s={w}x{h}:d=0.2", "-vf", "format=yuv420p", "-frames:v", "1",
             "-c:v", name, "-f", "null", "-"],
            capture_output=True, timeout=40, **_popen_kwargs())
        ok = r.returncode == 0
    except Exception:
        ok = False
    _ENCODER_CACHE[key] = ok
    return ok


def video_encoder_args(ffmpeg: str, codec: str, quality: str, w: int, h: int,
                       ext: str = ".mp4") -> List[str]:
    """Picks the fastest encoder that really works at THIS frame size
    (hardware H.264 stops at 4096 px, hardware H.265 at 8192 px)."""
    crf = {"high": 15, "standard": 18, "small": 22}.get(quality, 15)
    tag = ["-tag:v", "hvc1"] if ext in (".mp4", ".mov") else []
    big = w > 4096 or h > 4096
    want_h265 = codec == "h265" or big
    if want_h265 and w <= 8192 and h <= 8192 and _encoder_works(ffmpeg, "hevc_nvenc", w, h):
        return ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf + 2),
                "-b:v", "0"] + tag
    if codec == "h265":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", str(crf + 3)] + tag
    if not big and _encoder_works(ffmpeg, "h264_nvenc", w, h):
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(crf + 1), "-b:v", "0", "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "high"]


def grab_video_frame(path: str, frac: float = 0.35) -> Tuple["np.ndarray", float, Tuple[int, int], float]:
    """Returns (frame BGR, fps, (w,h), duration seconds)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open this video")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if n > 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * clamp(frac, 0.0, 0.95)))
    ok, frame = cap.read()
    if not ok and n > 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError("Cannot read this video")
    if not (1.0 < fps < 241.0):
        fps = 25.0
    dur = (n / fps) if n > 0 else 0.0
    return frame, fps, (w or frame.shape[1], h or frame.shape[0]), dur


# ============================================================================
#  TURBO  -  the 4th quality level, offered only for videos
#  (the three original levels and the original engine are not touched)
#   * a very small, very fast model (SPAN) with whole-frame processing
#   * the AI works on "key" frames; its detail is carried to the frames in
#     between with optical flow (60 fps frames are nearly identical, so the
#     result looks the same - and the picture is steadier, with less flicker)
#   * the app measures its own speed and picks how many frames sit between
#     key frames, so a clip finishes inside the time budget.  A fast graphics
#     card simply runs the AI on every frame.
# ============================================================================
_TURBO_HF = "https://huggingface.co/Phips"
MODELS["turbo2x"] = ModelSpec(
    "turbo2x", "Turbo",
    ("2xNomosUni_span_multijpg.safetensors", "2xNomosUni_span_multijpg.pth"),
    (f"{_TURBO_HF}/2xNomosUni_span_multijpg/resolve/main/2xNomosUni_span_multijpg.safetensors",
     f"{_GH}/2xNomosUni_span_multijpg/2xNomosUni_span_multijpg.pth"),
    0,
)
MODELS["turbo4x"] = ModelSpec(
    "turbo4x", "Turbo",
    ("4xNomosUni_span_multijpg.safetensors", "4xNomosUni_span_multijpg.pth"),
    (f"{_TURBO_HF}/4xNomosUni_span_multijpg/resolve/main/4xNomosUni_span_multijpg.safetensors",
     f"{_GH}/4xNomosUni_span_multijpg/4xNomosUni_span_multijpg.pth"),
    0,
)


class VideoEngine(Engine):
    """The same engine, tuned for the small video model.  Only used by Turbo."""

    def __init__(self, use_gpu: bool = True, gpu_index: int = -1):
        super().__init__(use_gpu, gpu_index)
        self.channels_last = False
        self._layout_tuned = False

    def ensure_model(self, spec: ModelSpec, status: Callable[[str, float], None],
                     cancel: Optional[threading.Event]) -> Path:
        path = find_model_file(spec)
        if path is not None:
            return path

        def prog(done: int, total: int) -> None:
            frac = (done / total) if total else 0.0
            status(f"Setting up Turbo (first time only)  {int(frac * 100)}%", frac)

        status("Setting up Turbo (first time only)", 0.0)
        last: Optional[Exception] = None
        for url in spec.urls:
            target = MODELS_DIR / Path(url).name
            try:
                download_file((url,), target, prog, cancel)
                return target
            except Cancelled:
                raise
            except Exception as e:
                last = e
        raise RuntimeError(f"Download failed: {last}")

    def load(self, spec: ModelSpec, status: Callable[[str, float], None],
             cancel: Optional[threading.Event]) -> None:
        before = self.key
        super().load(spec, status, cancel)
        if self.key != before:
            self.channels_last = False
            self._layout_tuned = False

    def auto_tile(self, video: bool = False) -> int:
        if self.heavy != 0:
            return super().auto_tile(video)
        if not self.on_gpu:
            return 384
        free = self.vram_free_gb()
        for need, tile in ((4.0, 2048), (3.0, 1536), (2.0, 1024), (1.3, 640)):
            if free >= need:
                return tile
        return 384

    def _run(self, piece: "torch.Tensor") -> "torch.Tensor":
        h, w = piece.shape[-2:]
        s = self.scale
        t = piece.to(torch.float16 if self.fp16 else torch.float32)
        ph, pw = self._pad_amounts(h, w)
        if ph or pw:
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            t = F.pad(t, (0, pw, 0, ph), mode=mode)
        if self.channels_last:
            t = t.contiguous(memory_format=torch.channels_last)
        out = self.desc(t)
        return out[:, :, :h * s, :w * s].clamp(0.0, 1.0)

    def upscale(self, src: "torch.Tensor", out_w: int, out_h: int,
                progress: Optional[Callable[[float], None]] = None,
                cancel: Optional[threading.Event] = None) -> "torch.Tensor":
        H, W = src.shape[-2:]
        if self.desc is not None and self.auto_tile(self.video_mode) >= max(H, W):
            res = self._infer(src)                      # whole frame in one go: no tiles, no blending
            if res.shape[-1] != out_w or res.shape[-2] != out_h:
                res = F.interpolate(res.float(), size=(out_h, out_w), mode="bicubic",
                                    align_corners=False, antialias=True).clamp_(0.0, 1.0)
            if progress:
                progress(1.0)
            return res
        return super().upscale(src, out_w, out_h, progress, cancel)

    def autotune_layout(self, sample: "torch.Tensor") -> None:
        """Once per video: pick the faster memory layout for the small model."""
        if self._layout_tuned or not self.on_gpu or self.desc is None:
            self._layout_tuned = True
            return
        self._layout_tuned = True
        try:
            x = sample[:, :, :384, :384].contiguous()
            dt = torch.float16 if self.fp16 else torch.float32
            mod = self.desc.model

            def bench(cl: bool) -> float:
                mod.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
                t = x.to(dt)
                t = t.contiguous(memory_format=torch.channels_last) if cl else t.contiguous()
                for _ in range(2):
                    self.desc(t)
                torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                for _ in range(4):
                    self.desc(t)
                torch.cuda.synchronize(self.device)
                return time.perf_counter() - t0

            a = bench(False)
            b = bench(True)
            self.channels_last = b < a * 0.93
            mod.to(memory_format=torch.channels_last if self.channels_last else torch.contiguous_format)
            log.info("Turbo layout test: normal %.1f ms, channels-last %.1f ms -> %s", a * 250, b * 250,
                     "channels-last" if self.channels_last else "normal")
        except Exception as e:
            log.warning("Turbo layout test skipped: %s", e)
            self.channels_last = False
            try:
                self.desc.model.to(memory_format=torch.contiguous_format)
            except Exception:
                pass


_VGRID: Dict[Tuple[int, int, str], "torch.Tensor"] = {}


def base_grid(H: int, W: int, dev) -> "torch.Tensor":
    key = (H, W, str(dev))
    g = _VGRID.get(key)
    if g is None:
        if len(_VGRID) > 2:
            _VGRID.clear()
        ys = torch.linspace(-1 + 1.0 / H, 1 - 1.0 / H, H, device=dev)
        xs = torch.linspace(-1 + 1.0 / W, 1 - 1.0 / W, W, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.stack([gx, gy], -1).unsqueeze(0)
        _VGRID[key] = g
    return g


def flow_to_grid(flow: "np.ndarray", H: int, W: int, dev) -> "torch.Tensor":
    """Optical flow (pixels, small size) -> sampling offsets for grid_sample at (H, W)."""
    t = torch.from_numpy(flow).to(dev).permute(2, 0, 1).unsqueeze(0)
    sx, sy = W / float(flow.shape[1]), H / float(flow.shape[0])
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    t = torch.cat([t[:, 0:1] * (sx * 2.0 / W), t[:, 1:2] * (sy * 2.0 / H)], dim=1)
    return t.permute(0, 2, 3, 1)


def gblur_c(x: "torch.Tensor", sigma: float) -> "torch.Tensor":
    kh, kv, r = _gauss_kernels(sigma, x.device)
    C = x.shape[1]
    mode = "reflect" if r < min(x.shape[-2:]) else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), kh.expand(C, 1, 1, -1).contiguous(), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), kv.expand(C, 1, -1, 1).contiguous(), groups=C)
    return x


def _resize_t(x: "torch.Tensor", ow: int, oh: int) -> "torch.Tensor":
    if x.shape[-1] == ow and x.shape[-2] == oh:
        return x
    return F.interpolate(x, size=(oh, ow), mode="bicubic", align_corners=False,
                         antialias=True).clamp(0.0, 1.0)


def video_ai_input(src: "torch.Tensor", ow: int, oh: int) -> "torch.Tensor":
    """Frames already close to the output size go to the fast 2x model at half the
    output size (cheap, and the AI re-creates the detail)."""
    if ow / float(src.shape[-1]) < 1.5:
        return _resize_t(src, max(2, ow // 2), max(2, oh // 2))
    return src


def video_dims(path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise IOError("Cannot read this video")
    return w, h


def turbo_spec(job: "Job", params: "EnhanceParams") -> ModelSpec:
    try:
        w, h = video_dims(job.path)
        _, _, s = plan_output(w, h, params.target, even=True)
        return MODELS["turbo2x"] if s <= 2.6 else MODELS["turbo4x"]
    except Exception:
        return MODELS["turbo2x"]


def _turbo_texture_y(y: "torch.Tensor", ys: "torch.Tensor", amount: int, sc: float) -> "torch.Tensor":
    """Natural-texture step without the grain (grain is added fresh on every frame)."""
    t = clamp(amount, 0, 100) / 100.0
    if t <= 0:
        return y
    band = gblur(ys, 1.1 * sc) - gblur(ys, 3.2 * sc)
    mean = gblur(y, 2.2 * sc)
    var = (gblur(y * y, 2.2 * sc) - mean * mean).clamp_min(0.0)
    flat = 1.0 - _tsmooth((torch.sqrt(var) - 0.004) / 0.03)
    return (y + band * flat * (0.95 * t)).clamp(0.0, 1.0)


def _turbo_finish(up: "torch.Tensor", src: "torch.Tensor", params: "EnhanceParams", sc: float,
                  noise_cache: List[float], fast: bool) -> "torch.Tensor":
    """Deblur / detail / texture (the 'static' look that is carried between frames)."""
    if not (params.remove_blur or params.detail > 0 or params.texture > 0):
        return up
    H, W = up.shape[-2:]
    ysrc = _luma(src)
    ys_full = None
    if params.texture > 0:
        if noise_cache and (noise_cache[1] % 15) != 0:
            noise_cache[1] += 1
        else:
            noise_cache[:] = [estimate_noise(ysrc), 1]
        ys_full = F.interpolate(ysrc, size=(H, W), mode="bicubic", align_corners=False)
    halo = int(clamp(48 * sc, 24, 128))
    rows = max(128, int(9e6 / W) - 2 * halo)
    res = torch.empty_like(up)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        a, b = max(0, r0 - halo), min(H, r1 + halo)
        band = up[:, :, a:b, :].float()
        y0 = _luma(band)
        y = y0
        if params.remove_blur:
            y = deblur_y(y, params.blur_strength, sc, fast)
        if params.detail > 0:
            y = detail_y(y, params.detail, sc)
        if params.texture > 0:
            y = _turbo_texture_y(y, ys_full[:, :, a:b, :], params.texture, sc)
        o = r0 - a
        n = r1 - r0
        delta = (y - y0)[:, :, o:o + n, :]
        res[:, :, r0:r1, :] = (band[:, :, o:o + n, :] + delta).clamp_(0.0, 1.0).to(up.dtype)
    return res


class TurboPipeline:
    BUDGET_X = 8.0        # processing seconds allowed per second of video (1 minute clip -> 8 minutes)
    N_MAX_60 = 8          # longest gap between AI key frames at 60 fps

    def __init__(self, engine: "Engine", params: "EnhanceParams", tf: Optional["TemporalGPU"],
                 src_w: int, ow: int, oh: int, fps: float, cancel: threading.Event):
        self.engine, self.params, self.tf, self.cancel = engine, params, tf, cancel
        self.dev = engine.device
        self.src_w, self.ow, self.oh = src_w, ow, oh
        self.target = self.BUDGET_X / max(1.0, fps)          # seconds we may spend on each source frame
        self.n_max = self.N_MAX_60 if fps >= 60 else max(2, int(round(self.N_MAX_60 * fps / 60.0)))
        self.N = 2                                           # frames per key-frame cycle (1 = AI on every frame)
        self.since = 0
        self.key: Optional[dict] = None
        self.t_ai: Optional[float] = None
        self.t_tr: Optional[float] = None
        self.nc: List[float] = []
        self.sc = _res_scale(ow, oh)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(7)

    @staticmethod
    def small_gray(bgr: "np.ndarray") -> "np.ndarray":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h = g.shape[0]
        if h > 540:
            g = cv2.resize(g, (int(g.shape[1] * 540 / h), 540), interpolation=cv2.INTER_AREA)
        return g

    def _src(self, bgr: "np.ndarray") -> "torch.Tensor":
        s = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to(self.dev)
        s = s.permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255.0)
        if self.params.remove_grain:
            s = gpu_denoise(s, self.params.grain_strength)
        return s

    def _ai_key(self, src: "torch.Tensor", gray: "np.ndarray") -> "torch.Tensor":
        eng = self.engine
        ai_in = video_ai_input(src, self.ow, self.oh)
        eng.autotune_layout(ai_in)
        up = eng.upscale(ai_in, self.ow, self.oh, None, self.cancel)
        res = _turbo_finish(up, src, self.params, self.sc, self.nc, self.N <= 1)
        del up
        base = _resize_t(src, self.ow, self.oh)
        self.key = {"gray": gray, "base": base.half(), "resid": (res.float() - base).half()}
        self.since = 1
        return res

    def _transfer(self, src: "torch.Tensor", gray: "np.ndarray") -> Tuple["torch.Tensor", float]:
        k = self.key
        H, W = self.oh, self.ow
        base = _resize_t(src, W, H)
        flow = self.dis.calc(gray, k["gray"], None)
        g2 = base_grid(H, W, self.dev) + flow_to_grid(flow, H, W, self.dev)
        rw = F.grid_sample(k["resid"].float(), g2, mode="bilinear", padding_mode="border", align_corners=False)
        wb = F.grid_sample(k["base"].float(), g2, mode="bilinear", padding_mode="border", align_corners=False)
        err = (wb - base).abs().mean(1, keepdim=True)
        conf = gblur(torch.exp(-torch.square(err / 0.05)), 2.0 * self.sc)
        cm = float(conf.mean())
        soft = (base - gblur_c(base, 1.1 * self.sc)) * 0.8       # plain sharpening where the detail cannot follow
        out = (base + rw * conf + soft * (1.0 - conf)).clamp(0.0, 1.0)
        self.since += 1
        return out, cm

    def _grain(self, res: "torch.Tensor") -> "torch.Tensor":
        """Fresh film grain on every frame (so it never looks 'stuck' to the picture)."""
        p = self.params
        t = clamp(p.texture, 0, 100) / 100.0
        if t <= 0:
            return res
        H, W = res.shape[-2:]
        x = res.float()
        y = _luma(x)
        n = torch.randn((1, 1, H, W), device=self.dev, dtype=torch.float32, generator=self.gen)
        f_up = self.ow / float(self.src_w)
        n = gblur(n, clamp(0.5 * (max(f_up, 1.0) ** 0.55), 0.6, 2.4) * self.sc)
        n = n / (n.std() + 1e-6)
        sigma_n = self.nc[0] if self.nc else 0.005
        keep = 1.0 - (0.75 * p.grain_strength / 100.0 if p.remove_grain else 0.0)
        amp = t * clamp(sigma_n * 0.9 + 0.004, 0.003, 0.02) * keep
        return (x + n * (0.30 + 0.70 * 4.0 * y * (1.0 - y)) * amp).clamp(0.0, 1.0)

    def _retune(self) -> None:
        if self.t_ai is None or self.t_tr is None:
            return
        if self.t_ai <= self.target:
            n = 1
        else:
            room = self.target - self.t_tr
            n = self.n_max if room <= 0.003 else int(math.ceil((self.t_ai - self.t_tr) / room))
        self.N = int(clamp(n, 1, self.n_max))
        log.info("Turbo: AI %.0f ms, carried %.0f ms, budget %.0f ms/frame -> AI on 1 of every %d frames",
                 self.t_ai * 1000, self.t_tr * 1000, self.target * 1000, self.N)

    @torch.inference_mode()
    def process(self, bgr: "np.ndarray", idx: int) -> "np.ndarray":
        t0 = time.perf_counter()
        gray = self.small_gray(bgr)
        src = self._src(bgr)
        res = None
        kind = "ai"
        if self.key is not None and idx >= 2 and self.since < self.N:
            if float(cv2.absdiff(gray, self.key["gray"]).mean()) <= 35.0:      # not a scene cut
                res, cm = self._transfer(src, gray)
                if cm >= 0.5:
                    kind = "tr"
                else:
                    res = None                                                  # moving too fast: use the AI
        if res is None:
            res = self._ai_key(src, gray)
            kind = "ai"
        res = self._grain(res)
        if kind == "ai" and self.N <= 1 and self.tf is not None:
            res = self.tf.apply(gblur(_luma(src), 1.2), res)
        out = (res.float() * 255.0 + 0.5).clamp_(0.0, 255.0).to(torch.uint8)
        out = out[0].permute(1, 2, 0).flip(-1).contiguous().cpu().numpy()
        dt = time.perf_counter() - t0
        if kind == "ai":
            if idx >= 1:                      # the very first frame includes warm-up
                self.t_ai = dt if self.t_ai is None else 0.8 * self.t_ai + 0.2 * dt
        else:
            self.t_tr = dt if self.t_tr is None else 0.8 * self.t_tr + 0.2 * dt
        if self.t_ai is not None and self.t_tr is not None and (idx == 2 or idx % 30 == 0):
            self._retune()
        return out


def process_video(engine: Engine, params: EnhanceParams, settings: Settings,
                  src: str, dst: Path,
                  report: Callable[[float, str], None],
                  cancel: threading.Event, preview_seconds: float = 0.0, turbo: bool = False) -> None:
    """Three stages run at the same time: read frames -> enhance on the GPU -> encode.
    The graphics card never waits for the disk or the encoder."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("Videos need a free helper called FFmpeg.\n\n"
                           "Install it once with:   winget install Gyan.FFmpeg\n"
                           "then restart YokoEnhancer.")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError("Cannot open this video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 < fps < 241.0):
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if preview_seconds > 0:
        limit = int(fps * preview_seconds)
        total = min(total, limit) if total > 0 else limit
    ow, oh, _ = plan_output(w, h, params.target, even=True)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps:.5f}", "-i", "-"]
    if preview_seconds > 0:
        cmd += ["-t", f"{preview_seconds:.3f}"]
    cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"]
    ext = dst.suffix.lower()
    cmd += video_encoder_args(ffmpeg, settings.video_codec, settings.video_quality, ow, oh, ext)
    cmd += ["-c:a", "aac", "-b:a", "256k", "-shortest"]
    if ext in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(dst)]

    err_file = tempfile.NamedTemporaryFile(prefix="yoko_ffmpeg_", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file, **_popen_kwargs())
    tf = TemporalGPU() if settings.steady_video else None
    vp = TurboPipeline(engine, params, tf, w, ow, oh, fps, cancel) if turbo else None

    frames_q: "queue.Queue" = queue.Queue(maxsize=6)
    out_q: "queue.Queue" = queue.Queue(maxsize=4)
    stop = threading.Event()
    werr: List[Optional[BaseException]] = [None]

    def reader() -> None:
        n = 0
        try:
            while not stop.is_set():
                if total > 0 and n >= total:
                    break
                ok, fr = cap.read()
                if not ok:
                    break
                while not stop.is_set():
                    try:
                        frames_q.put(fr, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                n += 1
        finally:
            while not stop.is_set():
                try:
                    frames_q.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def writer() -> None:
        try:
            while True:
                item = out_q.get()
                if item is None:
                    break
                proc.stdin.write(memoryview(item))
        except BaseException as e:      # noqa: BLE001
            werr[0] = e

    rt = threading.Thread(target=reader, daemon=True)
    wt = threading.Thread(target=writer, daemon=True)
    rt.start()
    wt.start()
    engine.set_video_mode(True)

    def ffmpeg_error() -> str:
        try:
            err_file.flush()
            return Path(err_file.name).read_text(errors="ignore")[-600:].strip() or "encoder stopped"
        except Exception:
            return "encoder stopped"

    done = 0
    failed: Optional[str] = None
    nc: List[float] = []
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            if werr[0] is not None:
                failed = ffmpeg_error()
                break
            try:
                frame = frames_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            if vp is not None:
                out = vp.process(frame, done)
            else:
                out = enhance_frame(engine, frame, params, ow, oh, None, cancel, seed=done,
                                    temporal=tf, video=True, noise_cache=nc)
            engine.check_finite = False
            while True:
                if cancel.is_set():
                    raise Cancelled()
                if werr[0] is not None:
                    break
                try:
                    out_q.put(out, timeout=0.2)
                    break
                except queue.Full:
                    continue
            done += 1
            report((done / total) if total > 0 else 0.5)
        if failed is None:
            while werr[0] is None:
                try:
                    out_q.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
            wt.join(timeout=600)
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait(timeout=600)
            if werr[0] is not None:
                failed = ffmpeg_error()
            elif proc.returncode not in (0, None):
                failed = ffmpeg_error()
            elif done == 0:
                failed = "No frames could be read from this video"
    except Cancelled:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        stop.set()
        engine.set_video_mode(False)
        engine.check_finite = True
        rt.join(timeout=3)
        cap.release()
        try:
            err_file.close()
            os.unlink(err_file.name)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    if failed:
        raise RuntimeError(f"Video could not be saved.\n{failed}")


# ============================================================================
#  JOBS + WORKER THREADS
# ============================================================================
class EtaTracker:
    """Real time-left estimate: measured from how fast work is *actually* going
    over the last ~40 seconds, smoothed, and counted down between updates."""

    def __init__(self, window: float = 40.0) -> None:
        self.window = window
        self.reset()

    def reset(self) -> None:
        self.samples: List[Tuple[float, float]] = []
        self._shown: Optional[float] = None
        self._last = 0.0

    def update(self, frac: float) -> None:
        now = time.time()
        if self.samples and frac < self.samples[-1][1] - 1e-6:
            self.reset()
        self.samples.append((now, frac))
        while len(self.samples) > 3 and now - self.samples[0][0] > self.window:
            self.samples.pop(0)

    def eta(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        t0, f0 = self.samples[0]
        t1, f1 = self.samples[-1]
        if t1 - t0 < 1.0 or f1 - f0 <= 1e-5:
            return None
        raw = (1.0 - f1) / ((f1 - f0) / (t1 - t0))
        now = time.time()
        if self._shown is None:
            self._shown = raw
        else:
            self._shown = max(0.0, self._shown - (now - self._last))
            self._shown = self._shown * 0.55 + raw * 0.45
        self._last = now
        return self._shown


@dataclass
class Job:
    path: str
    kind: str                       # image | video
    name: str
    status: str = "queued"          # queued | working | done | error
    progress: float = 0.0
    message: str = ""
    out_path: str = ""
    saved_path: str = ""            # set once the user presses Download
    size: Tuple[int, int] = (0, 0)
    duration: float = 0.0
    thumb: Optional[QImage] = None


def make_output_path(job: Job, settings: Settings, params: EnhanceParams) -> Path:
    src = Path(job.path)
    folder = WORK_DIR
    folder.mkdir(parents=True, exist_ok=True)
    label = target_label(params.target)
    if job.kind == "video":
        ext = "." + (settings.video_format if settings.video_format in ("mp4", "mov", "mkv") else "mp4")
    else:
        ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp", "tiff": ".tif"}.get(settings.output_format, ".png")
    return unique_path(folder / f"{src.stem}_Yoko_{label}{ext}")


class EnhanceWorker(QThread):
    status = Signal(str)
    overall = Signal(float)
    jobStarted = Signal(int)
    jobProgress = Signal(int, float)
    jobDone = Signal(int, str, object)       # index, output path, result array (images) or None
    jobFailed = Signal(int, str)
    fatal = Signal(str)
    allDone = Signal(bool)                   # True when cancelled

    def __init__(self, engine: Engine, jobs: List[Tuple[int, Job]], params: EnhanceParams,
                 settings: Settings, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.jobs = jobs
        self.params = params
        self.settings = Settings(**asdict(settings))
        self.cancel_event = threading.Event()
        self.eta = EtaTracker()
        self.turbo_engine: Optional["VideoEngine"] = None
        self._cur_engine = engine

    def stop(self) -> None:
        self.cancel_event.set()

    def _is_turbo(self, job: Job) -> bool:
        return self.params.quality == "turbo" and job.kind == "video"

    def _photo_quality(self) -> str:
        q = self.params.quality
        if q in ("maximum", "balanced", "fast"):
            return q
        pq = self.settings.photo_quality
        return pq if pq in ("maximum", "balanced", "fast") else "maximum"

    def _engine_for(self, job: Job) -> Engine:
        if not self._is_turbo(job):
            return self.engine
        if self.turbo_engine is None:
            self.engine.desc = None                     # give the graphics memory to Turbo
            self.engine.key = None
            self.engine.free_memory()
            self.turbo_engine = VideoEngine(self.settings.use_gpu, self.settings.gpu_index)
        return self.turbo_engine

    def _spec_for(self, job: Job) -> ModelSpec:
        if self._is_turbo(job):
            return turbo_spec(job, self.params)
        return MODELS[self._photo_quality()]

    # -------------------------------------------------------------------- run
    def run(self) -> None:
        cancelled = False
        try:
            spec = self._spec_for(self.jobs[0][1])

            def dl_status(text: str, frac: float) -> None:
                self.status.emit(text)
                self.overall.emit(frac)

            self._engine_for(self.jobs[0][1]).load(spec, dl_status, self.cancel_event)
            self.eta.reset()
        except Cancelled:
            self.allDone.emit(True)
            return
        except Exception as e:
            log.error("Model load failed: %s\n%s", e, traceback.format_exc())
            self.fatal.emit(self._friendly_model_error(e))
            self.allDone.emit(False)
            return

        n = len(self.jobs)
        for pos, (idx, job) in enumerate(self.jobs):
            if self.cancel_event.is_set():
                cancelled = True
                break
            self.jobStarted.emit(idx)
            t0 = time.time()
            eng_j = self._engine_for(job)
            spec_j = self._spec_for(job)
            if eng_j.key != spec_j.key:                 # only differs when Turbo and photos are mixed
                try:
                    eng_j.load(spec_j, lambda t, f: (self.status.emit(t), self.overall.emit(f)),
                               self.cancel_event)
                except Cancelled:
                    cancelled = True
                    self.jobFailed.emit(idx, "__cancelled__")
                    break
                except Exception as e:
                    log.error("Model load failed: %s", e)
                    self.jobFailed.emit(idx, self._friendly_model_error(e))
                    continue
            self._cur_engine = eng_j
            try:
                if job.kind == "video":
                    self._do_video(pos, n, idx, job)
                else:
                    self._do_image(pos, n, idx, job, t0)
            except Cancelled:
                cancelled = True
                self.jobFailed.emit(idx, "__cancelled__")
                break
            except Exception as e:
                log.error("Job failed: %s\n%s", e, traceback.format_exc())
                self.jobFailed.emit(idx, str(e))
        self.engine.free_memory()
        if self.turbo_engine is not None:
            self.turbo_engine.free_memory()
        self.allDone.emit(cancelled)

    @staticmethod
    def _friendly_model_error(e: Exception) -> str:
        msg = str(e)
        if "Download failed" in msg or "urlopen" in msg or "Name or service" in msg:
            return ("The files needed for this quality level could not be downloaded.\n"
                    "Please check your internet connection and try again.\n\n"
                    f"Offline? Copy the file into:\n{MODELS_DIR}")
        if "nsupported" in msg or "architecture" in msg.lower():
            return msg + "\n\nPlease update the helper library:  pip install -U spandrel"
        return msg

    def _report(self, pos: int, n: int, idx: int, frac: float, label: str) -> None:
        frac = clamp(frac, 0.0, 1.0)
        overall = (pos + frac) / n
        self.jobProgress.emit(idx, frac)
        self.overall.emit(overall)
        self.eta.update(overall)
        eta = self.eta.eta()
        prefix = f"({pos + 1} of {n})  " if n > 1 else ""
        tail = f"  -  about {fmt_time(eta)} left" if eta is not None else "  -  estimating time..."
        self.status.emit(f"{prefix}{label}  {int(frac * 100)}%{tail}")

    # ------------------------------------------------------------------ image
    def _do_image(self, pos: int, n: int, idx: int, job: Job, t0: float) -> None:
        bgr, alpha = imread_any(job.path)
        h, w = bgr.shape[:2]
        ow, oh, _ = plan_output(w, h, self.params.target)
        label = ["Enhancing"]

        def prog(f: float) -> None:
            self._report(pos, n, idx, 0.04 + 0.86 * f, label[0])

        def stage(name: str) -> None:
            label[0] = name
            if name == "Finishing":
                self._report(pos, n, idx, 0.92, name)
            elif name == "Cleaning":
                self._report(pos, n, idx, 0.02, name)

        out = enhance_frame(self.engine, bgr, self.params, ow, oh, prog, self.cancel_event,
                            seed=1, stage=stage)
        a2 = None
        if alpha is not None:
            a2 = cv2.resize(alpha, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        self._report(pos, n, idx, 0.96, "Saving")
        dst = make_output_path(job, self.settings, self.params)
        imwrite_any(dst, out, a2, self.settings.output_format, self.settings.jpg_quality)
        self.jobDone.emit(idx, str(dst), out)

    # ------------------------------------------------------------------ video
    def _do_video(self, pos: int, n: int, idx: int, job: Job) -> None:
        dst = make_output_path(job, self.settings, self.params)

        def report(frac: float) -> None:
            self._report(pos, n, idx, frac, "Enhancing")

        try:
            process_video(self._cur_engine, self.params, self.settings, job.path, dst, report,
                          self.cancel_event, turbo=self._is_turbo(job))
        except Exception:
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
            raise
        self.jobDone.emit(idx, str(dst), None)


class PreviewWorker(QThread):
    """Enhances a small area around the centre of the view so the user can
    judge the result in seconds."""
    done = Signal(object, object, object)        # before crop, after crop, (x, y, w, h)
    failed = Signal(str)
    status = Signal(str)

    CROP = {"maximum": 288, "balanced": 352, "fast": 448}

    def __init__(self, engine: Engine, job: Job, params: EnhanceParams, center: Tuple[float, float],
                 parent=None):
        super().__init__(parent)
        self.engine = engine
        self.job = job
        self.params = params
        self.center = center
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            turbo = self.params.quality == "turbo" and self.job.kind == "video"
            if turbo:
                eng = VideoEngine(self.engine.on_gpu, self.engine._idx() if self.engine.on_gpu else -1)
                spec = turbo_spec(self.job, self.params)
            else:
                eng = self.engine
                spec = MODELS[self.params.quality if self.params.quality in ("maximum", "balanced", "fast")
                              else "maximum"]

            def st(text: str, frac: float) -> None:
                self.status.emit(text)

            eng.load(spec, st, self.cancel_event)
            if self.job.kind == "video":
                bgr, _, _, _ = grab_video_frame(self.job.path, 0.35)
            else:
                bgr, _ = imread_any(self.job.path)
            h, w = bgr.shape[:2]
            _, _, scale = plan_output(w, h, self.params.target)
            c = min(self.CROP.get(self.params.quality, 512), w, h)
            cx = int(clamp(self.center[0], 0.0, 1.0) * w)
            cy = int(clamp(self.center[1], 0.0, 1.0) * h)
            x0 = int(clamp(cx - c // 2, 0, w - c))
            y0 = int(clamp(cy - c // 2, 0, h - c))
            crop = np.ascontiguousarray(bgr[y0:y0 + c, x0:x0 + c])
            ow, oh = max(2, int(round(c * scale))), max(2, int(round(c * scale)))
            self.status.emit("Previewing")
            if turbo:
                eng.set_video_mode(True)
            try:
                out = enhance_frame(eng, crop, self.params, ow, oh, None, self.cancel_event, seed=1, video=turbo)
            finally:
                if turbo:
                    eng.set_video_mode(False)
                    eng.desc = None
                    eng.free_memory()
            self.done.emit(crop, out, (x0, y0, c, c))
        except Cancelled:
            self.failed.emit("")
        except Exception as e:
            log.error("Preview failed: %s\n%s", e, traceback.format_exc())
            self.failed.emit(EnhanceWorker._friendly_model_error(e))


# ============================================================================
#                              U S E R   I N T E R F A C E
#       liquid-glass  /  deep black  /  cream-white text   (all custom drawn)
# ============================================================================
# ----------------------------------------------------------------------------
#  Colour themes  (the glass, layout and everything else stay identical)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    key: str
    name: str
    text: Tuple[int, int, int]      # text colour
    rim: Tuple[int, int, int]       # light colour for glass edges and shine
    glass: Tuple[int, int, int]     # colour of the glass body itself
    accent: Tuple[int, int, int]    # sliders, switches, progress, primary button
    glow: Tuple[int, int, int]      # ambient light in the background
    bg: Tuple[int, int, int]        # base background
    body: float                     # how visible the glass body colour is
    blob: float                     # strength of the background light
    colored: bool                   # True = real colour theme


THEMES: List[Theme] = [
    Theme("ivory", "Ivory Noir", (244, 236, 216), (255, 248, 232), (255, 248, 232), (250, 242, 224),
          (255, 238, 205), (2, 2, 2), 1.0, 1.0, False),
    Theme("amethyst", "Amethyst", (244, 238, 255), (224, 206, 255), (172, 142, 236), (198, 172, 255),
          (150, 112, 236), (9, 6, 20), 2.0, 2.3, True),
    Theme("obsidian", "Obsidian", (230, 230, 233), (205, 207, 216), (255, 255, 255), (224, 226, 232),
          (170, 174, 186), (0, 0, 0), 0.30, 0.28, False),
    Theme("crimson", "Crimson", (255, 238, 236), (255, 196, 200), (212, 100, 116), (242, 146, 156),
          (204, 74, 94), (12, 4, 6), 1.9, 2.2, True),
    Theme("ocean", "Ocean", (236, 249, 253), (190, 230, 248), (108, 176, 222), (152, 208, 242),
          (82, 162, 222), (3, 10, 18), 1.9, 2.3, True),
    Theme("emerald", "Emerald", (238, 251, 243), (190, 240, 214), (98, 190, 152), (142, 224, 188),
          (72, 182, 138), (3, 11, 9), 1.9, 2.2, True),
    Theme("rose", "Rose", (255, 240, 246), (255, 208, 226), (240, 144, 180), (250, 178, 208),
          (240, 122, 172), (14, 5, 10), 1.9, 2.3, True),
    Theme("sunset", "Sunset", (255, 244, 230), (255, 214, 172), (240, 160, 100), (250, 188, 132),
          (238, 140, 72), (14, 7, 3), 1.9, 2.2, True),
]


class _Current:
    theme: Theme = THEMES[0]


CUR = _Current()


def find_theme(key: str) -> Theme:
    for t in THEMES:
        if t.key == key:
            return t
    return THEMES[0]


def set_theme(key: str) -> None:
    CUR.theme = find_theme(key)


def _mix(a: Tuple[int, int, int], b: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    return (int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f), int(a[2] + (b[2] - a[2]) * f))


def _rgba(c: Tuple[int, int, int], a: float) -> QColor:
    return QColor(c[0], c[1], c[2], int(clamp(a, 0, 255)))


def _rgba_glass(a: float) -> QColor:
    return _rgba(CUR.theme.accent, a)


def _rgba_body(a: float) -> QColor:
    return _rgba(CUR.theme.glass, a)


def _accent(a: float) -> QColor:
    return _rgba(CUR.theme.accent, a)


def _rgba_glow(a: float) -> QColor:
    return _rgba(CUR.theme.glow, a)


def _bg_mix(f: float, a: float) -> QColor:
    return _rgba(_mix(CUR.theme.bg, CUR.theme.glass, f), a)


def _knob_hi() -> QColor:
    return QColor(*_mix(CUR.theme.text, (255, 255, 255), 0.7))


def _knob_lo(f: float = 1.0) -> QColor:
    t = CUR.theme.text
    k = 0.62 + 0.26 * f
    return QColor(int(t[0] * k), int(t[1] * k), int(t[2] * k))


def _label_qss(alpha: int) -> str:
    t = CUR.theme.text
    return f"color: rgba({t[0]},{t[1]},{t[2]},{alpha}); background: transparent;"


def restyle_tree(root: QWidget) -> None:
    """Re-colour every label under root after a theme change and repaint."""
    for lab in root.findChildren(QLabel):
        a = lab.property("yk_alpha")
        if a is not None:
            lab.setStyleSheet(_label_qss(int(a)))
    for w in root.findChildren(QWidget):
        w.update()
    root.update()


def cream(a: int = 255) -> QColor:
    t = CUR.theme.text
    return QColor(t[0], t[1], t[2], int(clamp(a, 0, 255)))


def warm(a: int = 255) -> QColor:
    """warm white used for glass fills and highlights"""
    g = CUR.theme.rim
    return QColor(g[0], g[1], g[2], int(clamp(a, 0, 255)))


FONT_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "SF Pro Display", "Inter",
                 "Helvetica Neue", "Arial"]


def ui_font(px: float, weight: QFont.Weight = QFont.Weight.Normal, spacing: float = 0.0) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(int(px))
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def make_label(text: str, px: float = 13, weight: QFont.Weight = QFont.Weight.Normal,
               alpha: int = 255, spacing: float = 0.0, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setFont(ui_font(px, weight, spacing))
    lab.setProperty("yk_alpha", alpha)
    lab.setStyleSheet(_label_qss(alpha))
    lab.setWordWrap(wrap)
    lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
    return lab


_SB_TPL = """
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: rgba(@T@,55); border-radius: 3px; min-height: 36px; }
QScrollBar::handle:vertical:hover { background: rgba(@T@,110); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 2px; }
QScrollBar::handle:horizontal { background: rgba(@T@,55); border-radius: 3px; min-width: 36px; }
QScrollBar::handle:horizontal:hover { background: rgba(@T@,110); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
"""

_MENU_TPL = """
QMenu { background: #0b0b0c; color: rgb(@T@); border: 1px solid rgba(@T@,60);
        border-radius: 12px; padding: 6px; }
QMenu::item { padding: 8px 22px; border-radius: 8px; }
QMenu::item:selected { background: rgba(@T@,34); }
"""


def _tpl(t: str) -> str:
    return t.replace("@T@", "%d,%d,%d" % CUR.theme.text)


def scrollbar_qss() -> str:
    return _tpl(_SB_TPL)


def menu_qss() -> str:
    return _tpl(_MENU_TPL)


def scroll_qss() -> str:
    return ("QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }" + scrollbar_qss())


def tooltip_qss() -> str:
    return ("QToolTip { background: #0d0d0d; color: rgb(%d,%d,%d); border: 1px solid rgba(%d,%d,%d,70);"
            " padding: 5px 8px; border-radius: 6px; }" % (CUR.theme.text + CUR.theme.text))



# ----------------------------------------------------------------------------
#  Animation helper
# ----------------------------------------------------------------------------
class Tween(QObject):
    """Animated float 0..1 (or any range) that triggers a repaint."""
    changed = Signal()

    def __init__(self, parent: QObject, value: float = 0.0, duration: int = 220,
                 curve: QEasingCurve.Type = QEasingCurve.Type.OutCubic):
        super().__init__(parent)
        self.value = float(value)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(curve)
        self._anim.valueChanged.connect(self._on)

    def _on(self, v) -> None:
        self.value = float(v)
        self.changed.emit()

    def to(self, target: float, duration: Optional[int] = None) -> None:
        target = float(target)
        if abs(target - self.value) < 1e-4 and self._anim.state() != QVariantAnimation.State.Running:
            return
        self._anim.stop()
        if duration is not None:
            self._anim.setDuration(duration)
        self._anim.setStartValue(self.value)
        self._anim.setEndValue(target)
        self._anim.start()

    def set(self, v: float) -> None:
        self._anim.stop()
        self.value = float(v)
        self.changed.emit()


# ----------------------------------------------------------------------------
#  Light that follows the cursor over every glass surface
# ----------------------------------------------------------------------------
class LightField(QObject):
    _inst: Optional["LightField"] = None

    @classmethod
    def get(cls) -> "LightField":
        if cls._inst is None:
            cls._inst = LightField()
        return cls._inst

    def __init__(self) -> None:
        super().__init__()
        self.pos = QPointF(-9999, -9999)
        self.widgets: "weakref.WeakSet[QWidget]" = weakref.WeakSet()
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def register(self, w: QWidget) -> None:
        self.widgets.add(w)
        w._lf_near = False  # type: ignore[attr-defined]

    def local(self, w: QWidget) -> QPointF:
        return QPointF(w.mapFromGlobal(self.pos.toPoint()))

    def _tick(self) -> None:
        p = QPointF(QCursor.pos())
        if p == self.pos:
            return
        self.pos = p
        for w in list(self.widgets):
            try:
                if not w.isVisible():
                    continue
                lp = w.mapFromGlobal(p.toPoint())
                near = w.rect().adjusted(-280, -280, 280, 280).contains(lp)
                if near or getattr(w, "_lf_near", False):
                    w.update()
                w._lf_near = near  # type: ignore[attr-defined]
            except RuntimeError:
                pass


# ----------------------------------------------------------------------------
#  The liquid glass look
# ----------------------------------------------------------------------------
def paint_glass(p: QPainter, r: QRectF, radius: float, strength: float = 1.0,
                light: Optional[QPointF] = None, glow: float = 0.0, tint: float = 0.0) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)

    # body
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    th = CUR.theme
    bd = th.body
    g.setColorAt(0.0, _rgba_body((30 * strength + 16 * glow) * bd + 40 * tint))
    g.setColorAt(0.5, _rgba_body((11 * strength + 7 * glow) * bd + 24 * tint))
    g.setColorAt(1.0, _rgba_body((17 * strength + 7 * glow) * bd + 34 * tint))
    p.fillPath(path, g)
    if th.colored and tint > 0:
        ag = QLinearGradient(r.topLeft(), r.bottomLeft())
        ag.setColorAt(0.0, _accent(105 * tint))
        ag.setColorAt(1.0, _accent(48 * tint))
        p.fillPath(path, ag)

    p.save()
    p.setClipPath(path)
    # deep inner vignette -> gives depth, keeps it black
    vg = QRadialGradient(r.center(), max(r.width(), r.height()) * 0.72)
    vg.setColorAt(0.55, QColor(0, 0, 0, 0))
    vg.setColorAt(1.0, QColor(0, 0, 0, int(80 * strength)))
    p.fillRect(r, vg)
    # top specular sheen
    sh = min(r.height() * 0.55, 140.0)
    sp = QLinearGradient(r.topLeft(), QPointF(r.left(), r.top() + sh))
    sp.setColorAt(0.0, warm(int(38 * strength)))
    sp.setColorAt(1.0, warm(0))
    p.fillRect(QRectF(r.left(), r.top(), r.width(), sh), sp)
    # liquid caustic bottom-right
    cg = QRadialGradient(QPointF(r.right() - r.width() * 0.08, r.bottom() + 4),
                         max(60.0, min(r.width(), r.height()) * 0.9))
    cg.setColorAt(0.0, warm(int(22 * strength)))
    cg.setColorAt(1.0, warm(0))
    p.fillRect(r, cg)
    # cursor light
    if light is not None:
        lg = QRadialGradient(light, 250)
        lg.setColorAt(0.0, _rgba_glass(int(44 * strength + 30 * glow)))
        lg.setColorAt(1.0, _rgba_glass(0))
        p.fillRect(r, lg)
    p.restore()

    # rim
    rim = QLinearGradient(r.topLeft(), r.bottomRight())
    rim.setColorAt(0.00, warm(int(165 * strength)))
    rim.setColorAt(0.30, warm(int(38 * strength)))
    rim.setColorAt(0.70, warm(int(24 * strength)))
    rim.setColorAt(1.00, warm(int(105 * strength)))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QBrush(rim), 1.2))
    p.drawPath(path)
    if light is not None:
        rl = QRadialGradient(light, 190)
        rl.setColorAt(0.0, _rgba_glass(int(150 * strength + 60 * glow)))
        rl.setColorAt(1.0, _rgba_glass(0))
        p.setPen(QPen(QBrush(rl), 1.2))
        p.drawPath(path)
    inner = QPainterPath()
    inner.addRoundedRect(r.adjusted(1.7, 1.7, -1.7, -1.7), max(1.0, radius - 1.7), max(1.0, radius - 1.7))
    p.setPen(QPen(warm(int(15 * strength)), 1.0))
    p.drawPath(inner)
    p.restore()


class GlassPanel(QWidget):
    """A pane of liquid glass.  Put a layout on it and add widgets."""

    def __init__(self, radius: float = 24, strength: float = 1.0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.radius = radius
        self.strength = strength
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        LightField.get().register(self)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(0.6, 0.6, -0.6, -0.6)
        paint_glass(p, r, self.radius, self.strength, LightField.get().local(self))
        p.end()


# ----------------------------------------------------------------------------
#  Icons
# ----------------------------------------------------------------------------
def _spark_path(cx: float, cy: float, R: float, k: float = 0.13) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(cx, cy - R)
    path.quadTo(cx + k * R, cy - k * R, cx + R, cy)
    path.quadTo(cx + k * R, cy + k * R, cx, cy + R)
    path.quadTo(cx - k * R, cy + k * R, cx - R, cy)
    path.quadTo(cx - k * R, cy - k * R, cx, cy - R)
    path.closeSubpath()
    return path


def draw_icon(p: QPainter, name: str, r: QRectF, color: QColor, lw: float = 1.7) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(color, lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    cx, cy = r.center().x(), r.center().y()
    P = QPointF
    if name == "plus":
        p.drawLine(P(cx, y + h * .2), P(cx, y + h * .8))
        p.drawLine(P(x + w * .2, cy), P(x + w * .8, cy))
    elif name == "close":
        p.drawLine(P(x + w * .25, y + h * .25), P(x + w * .75, y + h * .75))
        p.drawLine(P(x + w * .75, y + h * .25), P(x + w * .25, y + h * .75))
    elif name == "min":
        p.drawLine(P(x + w * .22, cy + h * .08), P(x + w * .78, cy + h * .08))
    elif name == "max":
        p.drawRoundedRect(QRectF(x + w * .24, y + h * .24, w * .52, h * .52), 2, 2)
    elif name == "restore":
        p.drawRoundedRect(QRectF(x + w * .22, y + h * .34, w * .44, h * .44), 2, 2)
        pth = QPainterPath()
        pth.moveTo(x + w * .36, y + h * .30)
        pth.lineTo(x + w * .36, y + h * .22)
        pth.lineTo(x + w * .78, y + h * .22)
        pth.lineTo(x + w * .78, y + h * .64)
        pth.lineTo(x + w * .70, y + h * .64)
        p.drawPath(pth)
    elif name == "folder":
        pth = QPainterPath()
        pth.moveTo(x + w * .14, y + h * .30)
        pth.lineTo(x + w * .14, y + h * .74)
        pth.quadTo(x + w * .14, y + h * .80, x + w * .20, y + h * .80)
        pth.lineTo(x + w * .80, y + h * .80)
        pth.quadTo(x + w * .86, y + h * .80, x + w * .86, y + h * .74)
        pth.lineTo(x + w * .86, y + h * .38)
        pth.quadTo(x + w * .86, y + h * .32, x + w * .80, y + h * .32)
        pth.lineTo(x + w * .50, y + h * .32)
        pth.lineTo(x + w * .42, y + h * .22)
        pth.lineTo(x + w * .20, y + h * .22)
        pth.quadTo(x + w * .14, y + h * .22, x + w * .14, y + h * .30)
        p.drawPath(pth)
    elif name == "spark":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPath(_spark_path(cx - w * .06, cy + h * .04, w * .40))
        p.drawPath(_spark_path(x + w * .80, y + h * .22, w * .16))
    elif name == "eye":
        pth = QPainterPath()
        pth.moveTo(x + w * .10, cy)
        pth.quadTo(cx, y + h * .06, x + w * .90, cy)
        pth.quadTo(cx, y + h * .94, x + w * .10, cy)
        p.drawPath(pth)
        p.drawEllipse(r.center(), w * .13, w * .13)
    elif name == "stop":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(x + w * .26, y + h * .26, w * .48, h * .48), 3, 3)
    elif name == "gear":
        p.drawEllipse(r.center(), w * .17, w * .17)
        p.drawEllipse(r.center(), w * .32, w * .32)
        for i in range(8):
            a = math.radians(i * 45)
            p.drawLine(P(cx + math.cos(a) * w * .32, cy + math.sin(a) * w * .32),
                       P(cx + math.cos(a) * w * .42, cy + math.sin(a) * w * .42))
    elif name == "trash":
        p.drawLine(P(x + w * .22, y + h * .28), P(x + w * .78, y + h * .28))
        p.drawLine(P(cx - w * .10, y + h * .28), P(cx - w * .10, y + h * .20))
        p.drawLine(P(cx + w * .10, y + h * .28), P(cx + w * .10, y + h * .20))
        pth = QPainterPath()
        pth.moveTo(x + w * .28, y + h * .28)
        pth.lineTo(x + w * .32, y + h * .80)
        pth.lineTo(x + w * .68, y + h * .80)
        pth.lineTo(x + w * .72, y + h * .28)
        p.drawPath(pth)
    elif name == "download":
        pth = QPainterPath()
        pth.moveTo(x + w * .50, y + h * .18)
        pth.lineTo(x + w * .50, y + h * .62)
        pth.moveTo(x + w * .32, y + h * .46)
        pth.lineTo(x + w * .50, y + h * .64)
        pth.lineTo(x + w * .68, y + h * .46)
        pth.moveTo(x + w * .20, y + h * .80)
        pth.lineTo(x + w * .80, y + h * .80)
        p.drawPath(pth)
    elif name == "check":
        pth = QPainterPath()
        pth.moveTo(x + w * .22, y + h * .52)
        pth.lineTo(x + w * .42, y + h * .72)
        pth.lineTo(x + w * .78, y + h * .30)
        p.drawPath(pth)
    elif name == "film":
        p.drawRoundedRect(QRectF(x + w * .16, y + h * .22, w * .68, h * .56), 3, 3)
        for i in range(3):
            yy = y + h * (.32 + .16 * i)
            p.drawPoint(P(x + w * .24, yy))
            p.drawPoint(P(x + w * .76, yy))
    elif name == "image":
        p.drawRoundedRect(QRectF(x + w * .16, y + h * .22, w * .68, h * .56), 3, 3)
        pth = QPainterPath()
        pth.moveTo(x + w * .20, y + h * .70)
        pth.lineTo(x + w * .42, y + h * .46)
        pth.lineTo(x + w * .58, y + h * .62)
        pth.lineTo(x + w * .68, y + h * .52)
        pth.lineTo(x + w * .80, y + h * .68)
        p.drawPath(pth)
    elif name == "warn":
        p.drawLine(P(cx, y + h * .26), P(cx, y + h * .58))
        p.drawPoint(P(cx, y + h * .75))
    elif name == "arrows":
        p.drawLine(P(x + w * .12, cy), P(x + w * .88, cy))
        pth = QPainterPath()
        pth.moveTo(x + w * .30, cy - h * .16)
        pth.lineTo(x + w * .12, cy)
        pth.lineTo(x + w * .30, cy + h * .16)
        p.drawPath(pth)
        pth = QPainterPath()
        pth.moveTo(x + w * .70, cy - h * .16)
        pth.lineTo(x + w * .88, cy)
        pth.lineTo(x + w * .70, cy + h * .16)
        p.drawPath(pth)
    p.restore()


def make_app_icon() -> QIcon:
    icon = QIcon()
    for size in (256, 128, 64, 48, 32, 16):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(size * .04, size * .04, size * .92, size * .92)
        path = QPainterPath()
        path.addRoundedRect(r, size * .26, size * .26)
        bg = QLinearGradient(r.topLeft(), r.bottomRight())
        bg.setColorAt(0, QColor(30, 29, 27))
        bg.setColorAt(1, QColor(3, 3, 3))
        p.fillPath(path, bg)
        rim = QLinearGradient(r.topLeft(), r.bottomRight())
        rim.setColorAt(0, warm(190))
        rim.setColorAt(.5, warm(30))
        rim.setColorAt(1, warm(110))
        p.setPen(QPen(QBrush(rim), max(1.0, size * .015)))
        p.drawPath(path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(cream(245))
        p.drawPath(_spark_path(size * .47, size * .53, size * .30))
        p.drawPath(_spark_path(size * .76, size * .27, size * .11))
        p.end()
        icon.addPixmap(pm)
    return icon


# ----------------------------------------------------------------------------
#  Button
# ----------------------------------------------------------------------------
class GlassButton(QAbstractButton):
    def __init__(self, text: str = "", icon: str = "", primary: bool = False, height: int = 44,
                 icon_only: bool = False, parent: Optional[QWidget] = None, radius: Optional[float] = None):
        super().__init__(parent)
        self._text = text
        self._icon = icon
        self.primary = primary
        self.icon_only = icon_only
        self._radius = radius if radius is not None else height / 2.0
        self._hover = Tween(self, 0.0, 180)
        self._press = Tween(self, 0.0, 90)
        self._hover.changed.connect(lambda: self.update())
        self._press.changed.connect(lambda: self.update())
        self.setFixedHeight(height)
        if icon_only:
            self.setFixedWidth(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFont(ui_font(14, QFont.Weight.DemiBold))
        LightField.get().register(self)

    def setLabel(self, text: str, icon: Optional[str] = None) -> None:
        self._text = text
        if icon is not None:
            self._icon = icon
        self.update()

    def sizeHint(self) -> QSize:
        if self.icon_only:
            return QSize(self.height(), self.height())
        fm = QFontMetrics(self.font())
        w = fm.horizontalAdvance(self._text) + 44 + (28 if self._icon else 0)
        return QSize(w, self.height())

    def enterEvent(self, ev) -> None:
        self._hover.to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._hover.to(0.0)
        self._press.to(0.0)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev) -> None:
        self._press.to(1.0)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        self._press.to(0.0)
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        en = self.isEnabled()
        if not en:
            p.setOpacity(0.38)
        hv, pr = self._hover.value, self._press.value
        r = QRectF(self.rect()).adjusted(1.2, 1.2, -1.2, -1.2)
        sc = 1.0 - 0.02 * pr
        c = r.center()
        p.translate(c)
        p.scale(sc, sc)
        p.translate(-c)
        strength = 0.75 + 0.45 * hv + (0.35 if self.primary else 0.0)
        paint_glass(p, r, self._radius, strength, LightField.get().local(self) if en else None,
                    glow=hv, tint=0.55 if self.primary else 0.0)
        col = cream(255)
        fm = QFontMetrics(self.font())
        p.setFont(self.font())
        if self.icon_only:
            s = min(r.width(), r.height()) * 0.5
            draw_icon(p, self._icon, QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s), col)
        else:
            tw = fm.horizontalAdvance(self._text)
            isz = 20 if self._icon else 0
            gap = 8 if self._icon else 0
            total = tw + isz + gap
            x0 = r.center().x() - total / 2
            if self._icon:
                draw_icon(p, self._icon, QRectF(x0, r.center().y() - isz / 2, isz, isz), col)
            p.setPen(col)
            p.drawText(QRectF(x0 + isz + gap, r.top(), tw + 4, r.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)
        p.end()


class WindowButton(QAbstractButton):
    """min / max / close in the title bar"""

    def __init__(self, icon: str, danger: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._icon = icon
        self.danger = danger
        self._hover = Tween(self, 0.0, 140)
        self._hover.changed.connect(lambda: self.update())
        self.setFixedSize(40, 30)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def setIcon_(self, name: str) -> None:
        self._icon = name
        self.update()

    def enterEvent(self, ev) -> None:
        self._hover.to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._hover.to(0.0)
        super().leaveEvent(ev)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(3, 2, -3, -2)
        hv = self._hover.value
        if hv > 0.01:
            path = QPainterPath()
            path.addRoundedRect(r, 9, 9)
            if self.danger:
                p.fillPath(path, QColor(190, 70, 60, int(150 * hv)))
            else:
                p.fillPath(path, warm(int(30 * hv)))
        s = 16
        draw_icon(p, self._icon, QRectF(self.width() / 2 - s / 2, self.height() / 2 - s / 2, s, s),
                  cream(230), 1.5)
        p.end()


# ----------------------------------------------------------------------------
#  Slider
# ----------------------------------------------------------------------------
class GlassSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(self, label: str, value: int = 50, minimum: int = 0, maximum: int = 100,
                 suffix: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label = label
        self._min, self._max = minimum, maximum
        self._value = int(clamp(value, minimum, maximum))
        self._suffix = suffix
        self._drag = False
        self._hover = Tween(self, 0.0, 160)
        self._hover.changed.connect(lambda: self.update())
        self.setFixedHeight(46)
        self.setMinimumWidth(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def value(self) -> int:
        return self._value

    def setValue(self, v: int, emit: bool = False) -> None:
        v = int(clamp(v, self._min, self._max))
        if v != self._value:
            self._value = v
            self.update()
            if emit:
                self.valueChanged.emit(v)

    def _track(self) -> QRectF:
        return QRectF(8, self.height() - 17, self.width() - 16, 6)

    def _frac(self) -> float:
        return (self._value - self._min) / float(max(1, self._max - self._min))

    def _set_from_x(self, x: float) -> None:
        tr = self._track()
        f = clamp((x - tr.left()) / max(1.0, tr.width()), 0.0, 1.0)
        v = int(round(self._min + f * (self._max - self._min)))
        if v != self._value:
            self._value = v
            self.update()
            self.valueChanged.emit(v)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._drag = True
            self._set_from_x(ev.position().x())

    def mouseMoveEvent(self, ev) -> None:
        if self._drag:
            self._set_from_x(ev.position().x())

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = False

    def wheelEvent(self, ev) -> None:
        if self.isEnabled():
            step = 1 if ev.angleDelta().y() > 0 else -1
            self.setValue(self._value + step * 2, emit=True)
            ev.accept()
        else:
            ev.ignore()

    def enterEvent(self, ev) -> None:
        self._hover.to(1.0)

    def leaveEvent(self, ev) -> None:
        self._hover.to(0.0)

    def changeEvent(self, ev) -> None:
        if ev.type() == QtCore.QEvent.Type.EnabledChange:
            self.update()
        super().changeEvent(ev)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.38)
        p.setFont(ui_font(12.5))
        p.setPen(cream(200))
        p.drawText(QRectF(8, 0, self.width() - 80, 22), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._label)
        p.setFont(ui_font(12.5, QFont.Weight.DemiBold))
        p.setPen(cream(255))
        p.drawText(QRectF(self.width() - 80, 0, 72, 22), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   f"{self._value}{self._suffix}")
        tr = self._track()
        groove = QPainterPath()
        groove.addRoundedRect(tr, 3, 3)
        p.fillPath(groove, warm(24))
        p.setPen(QPen(warm(40), 1))
        p.drawPath(groove)
        fx = tr.left() + tr.width() * self._frac()
        fill = QPainterPath()
        fill.addRoundedRect(QRectF(tr.left(), tr.top(), max(6.0, fx - tr.left()), tr.height()), 3, 3)
        fg = QLinearGradient(tr.topLeft(), tr.topRight())
        fg.setColorAt(0, _accent(110))
        fg.setColorAt(1, _accent(235))
        p.fillPath(fill, fg)
        hv = self._hover.value
        kr = 8.0 + 1.5 * hv + (1.5 if self._drag else 0.0)
        kc = QPointF(fx, tr.center().y())
        glow = QRadialGradient(kc, kr * 2.6)
        glow.setColorAt(0, _accent(60 + 50 * hv))
        glow.setColorAt(1, _accent(0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(kc, kr * 2.6, kr * 2.6)
        kg = QRadialGradient(QPointF(kc.x() - kr * .3, kc.y() - kr * .4), kr * 1.6)
        kg.setColorAt(0, _knob_hi())
        kg.setColorAt(1, _knob_lo())
        p.setBrush(kg)
        p.setPen(QPen(warm(200), 1))
        p.drawEllipse(kc, kr, kr)
        p.end()


# ----------------------------------------------------------------------------
#  Switch
# ----------------------------------------------------------------------------
class GlassSwitch(QAbstractButton):
    def __init__(self, checked: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self._t = Tween(self, 1.0 if checked else 0.0, 200)
        self._t.changed.connect(lambda: self.update())
        self.toggled.connect(lambda c: self._t.to(1.0 if c else 0.0))
        self.setFixedSize(48, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            p.setOpacity(0.4)
        t = self._t.value
        r = QRectF(self.rect()).adjusted(1.5, 3, -1.5, -3)
        path = QPainterPath()
        path.addRoundedRect(r, r.height() / 2, r.height() / 2)
        base = QLinearGradient(r.topLeft(), r.bottomLeft())
        base.setColorAt(0, _accent(26 + 120 * t))
        base.setColorAt(1, _accent(12 + 70 * t))
        p.fillPath(path, base)
        p.setPen(QPen(warm(int(60 + 100 * t)), 1.1))
        p.drawPath(path)
        kr = r.height() / 2 - 3
        kx = lerp(r.left() + kr + 3, r.right() - kr - 3, t)
        kc = QPointF(kx, r.center().y())
        kg = QRadialGradient(QPointF(kc.x() - kr * .3, kc.y() - kr * .4), kr * 1.7)
        kg.setColorAt(0, _knob_hi())
        kg.setColorAt(1, _knob_lo(t))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(kg)
        p.drawEllipse(kc, kr, kr)
        p.end()


class ToggleSlider(QWidget):
    """Title + switch on the first row, strength slider underneath."""
    changed = Signal(bool, int)

    def __init__(self, title: str, hint: str, checked: bool, value: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(8, 0, 4, 0)
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addWidget(make_label(title, 14.5, QFont.Weight.DemiBold))
        if hint:
            col.addWidget(make_label(hint, 11.5, alpha=135))
        top.addLayout(col)
        top.addStretch(1)
        self.switch = GlassSwitch(checked)
        top.addWidget(self.switch)
        lay.addLayout(top)
        self.slider = GlassSlider("Strength", value, 0, 100, "%")
        self.slider.setEnabled(checked)
        lay.addWidget(self.slider)
        self.switch.toggled.connect(self._on_toggle)
        self.slider.valueChanged.connect(lambda v: self.changed.emit(self.switch.isChecked(), v))

    def _on_toggle(self, c: bool) -> None:
        self.slider.setEnabled(c)
        self.changed.emit(c, self.slider.value())


# ----------------------------------------------------------------------------
#  Segmented chooser
# ----------------------------------------------------------------------------
class Segmented(QWidget):
    changed = Signal(str)

    def __init__(self, options: List[Tuple[str, str, str]], value: str, height: int = 42,
                 parent: Optional[QWidget] = None):
        """options: (key, label, sublabel)"""
        super().__init__(parent)
        self.options = options
        self.keys = [o[0] for o in options]
        self._idx = self.keys.index(value) if value in self.keys else 0
        self._pos = Tween(self, float(self._idx), 260, QEasingCurve.Type.OutBack)
        self._pos.changed.connect(lambda: self.update())
        self._hover_idx = -1
        self.setFixedHeight(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.has_sub = any(o[2] for o in options)
        LightField.get().register(self)

    def value(self) -> str:
        return self.keys[self._idx]

    def setValue(self, key: str, emit: bool = False) -> None:
        if key in self.keys:
            i = self.keys.index(key)
            if i != self._idx:
                self._idx = i
                self._pos.to(float(i))
                if emit:
                    self.changed.emit(key)

    def _index_at(self, x: float) -> int:
        n = len(self.options)
        seg = (self.width() - 8) / float(n)
        return int(clamp((x - 4) // seg, 0, n - 1))

    def mouseMoveEvent(self, ev) -> None:
        i = self._index_at(ev.position().x())
        if i != self._hover_idx:
            self._hover_idx = i
            self.update()

    def leaveEvent(self, ev) -> None:
        self._hover_idx = -1
        self.update()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            i = self._index_at(ev.position().x())
            if i != self._idx:
                self._idx = i
                self._pos.to(float(i))
                self.changed.emit(self.keys[i])

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.8, 0.8, -0.8, -0.8)
        rad = min(r.height() / 2, 22)
        trough = QPainterPath()
        trough.addRoundedRect(r, rad, rad)
        p.fillPath(trough, QColor(0, 0, 0, 90))
        p.setPen(QPen(warm(34), 1))
        p.drawPath(trough)
        n = len(self.options)
        seg = (r.width() - 6) / float(n)
        hl = QRectF(r.left() + 3 + seg * self._pos.value, r.top() + 3, seg, r.height() - 6)
        paint_glass(p, hl, max(4.0, rad - 3), 1.15, LightField.get().local(self), glow=0.6, tint=0.35)
        for i, (key, label, sub) in enumerate(self.options):
            cell = QRectF(r.left() + 3 + seg * i, r.top(), seg, r.height())
            sel = (i == self._idx)
            a = 255 if sel else (205 if i == self._hover_idx else 150)
            if self.has_sub and sub:
                p.setFont(ui_font(13.5, QFont.Weight.DemiBold))
                p.setPen(cream(a))
                p.drawText(QRectF(cell.left(), cell.top() + 6, cell.width(), cell.height() / 2),
                           Qt.AlignmentFlag.AlignCenter, label)
                p.setFont(ui_font(10.5))
                p.setPen(cream(int(a * 0.6)))
                p.drawText(QRectF(cell.left(), cell.center().y() + 1, cell.width(), cell.height() / 2 - 5),
                           Qt.AlignmentFlag.AlignCenter, sub)
            else:
                p.setFont(ui_font(13.5, QFont.Weight.DemiBold if sel else QFont.Weight.Medium))
                p.setPen(cream(a))
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, label)
        p.end()


# ----------------------------------------------------------------------------
#  Progress bar
# ----------------------------------------------------------------------------
class GlassProgress(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._value = 0.0
        self._shown = Tween(self, 0.0, 260)
        self._shown.changed.connect(lambda: self.update())
        self._active = False
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._step)
        self.setFixedHeight(10)

    def setActive(self, a: bool) -> None:
        self._active = a
        if a:
            self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def setValue(self, v: float) -> None:
        """0..1 ; negative = busy / indeterminate"""
        self._value = v
        if v >= 0:
            self._shown.to(clamp(v, 0.0, 1.0))
        self.update()

    def _step(self) -> None:
        self._phase = (self._phase + 0.018) % 1.0
        self.update()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        rad = r.height() / 2
        groove = QPainterPath()
        groove.addRoundedRect(r, rad, rad)
        p.fillPath(groove, QColor(0, 0, 0, 120))
        p.setPen(QPen(warm(38), 1))
        p.drawPath(groove)
        p.save()
        p.setClipPath(groove)
        if self._value < 0:
            w = r.width() * 0.32
            x = r.left() - w + (r.width() + w) * self._phase
            g = QLinearGradient(QPointF(x, 0), QPointF(x + w, 0))
            g.setColorAt(0, _accent(0))
            g.setColorAt(0.5, _accent(230))
            g.setColorAt(1, _accent(0))
            p.fillRect(QRectF(x, r.top(), w, r.height()), g)
        else:
            fw = max(rad * 2, r.width() * self._shown.value) if self._shown.value > 0.001 else 0
            if fw > 0:
                fr = QRectF(r.left(), r.top(), fw, r.height())
                g = QLinearGradient(fr.topLeft(), fr.topRight())
                g.setColorAt(0, _accent(120))
                g.setColorAt(1, _accent(240))
                p.fillRect(fr, g)
                if self._active:
                    sw = 90.0
                    sx = r.left() + (fw + sw) * self._phase - sw
                    sg = QLinearGradient(QPointF(sx, 0), QPointF(sx + sw, 0))
                    sg.setColorAt(0, QColor(255, 255, 255, 0))
                    sg.setColorAt(0.5, QColor(255, 255, 255, 120))
                    sg.setColorAt(1, QColor(255, 255, 255, 0))
                    p.fillRect(fr.intersected(QRectF(sx, r.top(), sw, r.height())), sg)
        p.restore()
        p.end()


# ----------------------------------------------------------------------------
#  Before / After viewer  (zoom, pan, draggable split)
# ----------------------------------------------------------------------------
class ImageLevels:
    """Image + lazily built half-size copies so zoomed-out drawing stays fast."""

    def __init__(self, img: QImage):
        self.levels: List[QImage] = [img]
        self.w = img.width()
        self.h = img.height()

    def get(self, scale: float) -> Tuple[QImage, float]:
        k = 0
        if scale < 1.0:
            k = int(clamp(math.floor(math.log2(1.0 / max(scale, 1e-4))), 0, 7))
        while len(self.levels) <= k:
            prev = self.levels[-1]
            if prev.width() <= 64 or prev.height() <= 64:
                break
            self.levels.append(prev.scaled(max(1, prev.width() // 2), max(1, prev.height() // 2),
                                           Qt.AspectRatioMode.IgnoreAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
        k = min(k, len(self.levels) - 1)
        lvl = self.levels[k]
        return lvl, lvl.width() / float(self.w)


class CompareView(QWidget):
    openRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.before: Optional[ImageLevels] = None
        self.after: Optional[ImageLevels] = None
        self.split = 0.5
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._drag: Optional[str] = None
        self._last = QPointF()
        self._cache: Optional[QPixmap] = None
        self._dirty = True
        self._busy: Optional[str] = None
        self._spin = 0.0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(33)
        self._spin_timer.timeout.connect(self._spin_step)
        self._hint = Tween(self, 0.0, 300)
        self._hint.changed.connect(lambda: self.update())
        self._empty_hover = Tween(self, 0.0, 200)
        self._empty_hover.changed.connect(self._invalidate)
        self._over_pill = False
        self.setMouseTracking(True)
        self.setMinimumSize(420, 300)
        self.setAcceptDrops(False)

    # ------------------------------------------------------------- public API
    def has_media(self) -> bool:
        return self.before is not None or self.after is not None

    def has_result(self) -> bool:
        return self.after is not None

    def set_before(self, bgr: "np.ndarray") -> None:
        self.before = ImageLevels(bgr_to_qimage(bgr))
        self.after = None
        self.reset_view()

    def set_after(self, bgr: Optional["np.ndarray"]) -> None:
        self.after = ImageLevels(bgr_to_qimage(bgr)) if bgr is not None else None
        self._invalidate()

    def set_pair(self, before_bgr: "np.ndarray", after_bgr: "np.ndarray") -> None:
        self.before = ImageLevels(bgr_to_qimage(before_bgr))
        self.after = ImageLevels(bgr_to_qimage(after_bgr))
        self.reset_view()

    def clear(self) -> None:
        self.before = None
        self.after = None
        self.reset_view()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.split = 0.5
        self._invalidate()

    def set_busy(self, text: Optional[str]) -> None:
        self._busy = text
        if text is not None and not self._spin_timer.isActive():
            self._spin_timer.start()
        elif text is None:
            self._spin_timer.stop()
        self.update()

    def center_norm(self) -> Tuple[float, float]:
        """Centre of the current view in 0..1 image coordinates."""
        if not self.has_media():
            return 0.5, 0.5
        dest, _, _ = self._dest()
        cx = (self.width() / 2 - dest.left()) / max(1.0, dest.width())
        cy = (self.height() / 2 - dest.top()) / max(1.0, dest.height())
        return clamp(cx, 0.0, 1.0), clamp(cy, 0.0, 1.0)

    # --------------------------------------------------------------- geometry
    def _dims(self) -> Tuple[int, int]:
        src = self.before or self.after
        return (src.w, src.h) if src else (1, 1)

    def _dest(self) -> Tuple[QRectF, float, float]:
        iw, ih = self._dims()
        vw, vh = float(self.width()), float(self.height())
        pad = 14.0
        fit = max(1e-4, min((vw - 2 * pad) / iw, (vh - 2 * pad) / ih))
        sc = fit * self.zoom
        w, h = iw * sc, ih * sc
        return QRectF((vw - w) / 2 + self.pan.x(), (vh - h) / 2 + self.pan.y(), w, h), sc, fit

    def _clamp_pan(self) -> None:
        iw, ih = self._dims()
        vw, vh = float(self.width()), float(self.height())
        _, sc, _ = self._dest()
        w, h = iw * sc, ih * sc
        mx = max(0.0, (w - vw) / 2 + 90) if w > vw else 0.0
        my = max(0.0, (h - vh) / 2 + 90) if h > vh else 0.0
        self.pan = QPointF(clamp(self.pan.x(), -mx, mx), clamp(self.pan.y(), -my, my))

    def _invalidate(self) -> None:
        self._dirty = True
        self.update()

    def _spin_step(self) -> None:
        self._spin = (self._spin + 9.0) % 360.0
        self.update()

    def _split_x(self, dest: QRectF) -> float:
        left = max(0.0, dest.left())
        right = min(float(self.width()), dest.right())
        x = self.split * self.width()
        return clamp(x, left + 2, right - 2)

    # ------------------------------------------------------------------ events
    def resizeEvent(self, ev) -> None:
        self._clamp_pan()
        self._invalidate()
        super().resizeEvent(ev)

    def enterEvent(self, ev) -> None:
        self._hint.to(1.0)
        if not self.has_media():
            self._empty_hover.to(1.0)

    def leaveEvent(self, ev) -> None:
        self._hint.to(0.0)
        self._empty_hover.to(0.0)

    def mousePressEvent(self, ev) -> None:
        if not self.has_media():
            if ev.button() == Qt.MouseButton.LeftButton:
                self.openRequested.emit()
            return
        self._last = ev.position()
        if ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag = "pan"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif ev.button() == Qt.MouseButton.LeftButton:
            if self.after is not None:
                self._drag = "split"
                self._move_split(ev.position().x())
            elif self.zoom > 1.01:
                self._drag = "pan"
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _move_split(self, x: float) -> None:
        dest, _, _ = self._dest()
        left = max(0.0, dest.left())
        right = min(float(self.width()), dest.right())
        x = clamp(x, left + 2, right - 2)
        self.split = x / max(1.0, self.width())
        self._invalidate()

    def mouseMoveEvent(self, ev) -> None:
        if self._drag == "split":
            self._move_split(ev.position().x())
        elif self._drag == "pan":
            d = ev.position() - self._last
            self._last = ev.position()
            self.pan = QPointF(self.pan.x() + d.x(), self.pan.y() + d.y())
            self._clamp_pan()
            self._invalidate()
        else:
            if not self.has_media():
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif self.after is not None:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom > 1.01 else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, ev) -> None:
        if self.has_media():
            self.zoom = 1.0
            self.pan = QPointF(0, 0)
            self._invalidate()

    def wheelEvent(self, ev) -> None:
        if not self.has_media():
            return
        f = 1.18 ** (ev.angleDelta().y() / 120.0)
        new_zoom = clamp(self.zoom * f, 1.0, 48.0)
        f = new_zoom / self.zoom
        if abs(f - 1.0) < 1e-6:
            return
        c = QPointF(self.width() / 2.0, self.height() / 2.0)
        m = ev.position()
        self.pan = m - (m - (c + self.pan)) * f - c
        self.zoom = new_zoom
        self._clamp_pan()
        self._invalidate()
        ev.accept()

    # ---------------------------------------------------------------- drawing
    def _draw_level(self, p: QPainter, lv: ImageLevels, dest: QRectF, clip: QRectF) -> None:
        inter = dest.intersected(clip)
        if inter.width() <= 0 or inter.height() <= 0:
            return
        scale = dest.width() / float(lv.w)
        img, f = lv.get(scale)
        k = f / max(scale, 1e-9)           # level pixels per screen pixel
        sx = (inter.left() - dest.left()) / dest.width() * img.width()
        sy = (inter.top() - dest.top()) / dest.height() * img.height()
        sw = inter.width() / dest.width() * img.width()
        sh = inter.height() / dest.height() * img.height()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 3.0)
        p.drawImage(inter, img, QRectF(sx, sy, sw, sh))
        _ = k

    def _pill(self, p: QPainter, text: str, x: float, y: float, align_right: bool = False,
              icon: str = "") -> QRectF:
        p.setFont(ui_font(12.5, QFont.Weight.DemiBold))
        fm = QFontMetrics(p.font())
        tw = fm.horizontalAdvance(text)
        w = tw + 26 + (18 if icon else 0)
        h = 28.0
        r = QRectF(x - w if align_right else x, y, w, h)
        path = QPainterPath()
        path.addRoundedRect(r, h / 2, h / 2)
        p.fillPath(path, QColor(0, 0, 0, 150))
        p.setPen(QPen(warm(70), 1))
        p.drawPath(path)
        tx = r.left() + 13
        if icon:
            draw_icon(p, icon, QRectF(tx - 2, r.center().y() - 8, 16, 16), cream(230), 1.4)
            tx += 18
        p.setPen(cream(245))
        p.drawText(QRectF(tx, r.top(), tw + 4, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        return r

    def _render(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        W, H = float(self.width()), float(self.height())
        full = QRectF(0, 0, W, H)
        clip = QPainterPath()
        clip.addRoundedRect(full, 20, 20)
        p.setClipPath(clip)
        p.fillRect(full, QColor(0, 0, 0, 175))

        if not self.has_media():
            self._render_empty(p, full)
            return

        dest, sc, _ = self._dest()
        # checkerboard under transparent / letterboxed area is not needed - plain black
        if self.after is None and self.before is not None:
            self._draw_level(p, self.before, dest, full)
        else:
            sx = self._split_x(dest)
            self._draw_level(p, self.before, dest, QRectF(0, 0, sx, H))          # type: ignore[arg-type]
            self._draw_level(p, self.after, dest, QRectF(sx, 0, W - sx, H))      # type: ignore[arg-type]
            # divider
            vis_top, vis_bot = max(0.0, dest.top()), min(H, dest.bottom())
            glow = QLinearGradient(QPointF(sx - 7, 0), QPointF(sx + 7, 0))
            glow.setColorAt(0, _accent(0))
            glow.setColorAt(0.5, _accent(70))
            glow.setColorAt(1, _accent(0))
            p.fillRect(QRectF(sx - 7, vis_top, 14, vis_bot - vis_top), glow)
            p.setPen(QPen(_accent(235), 1.6))
            p.drawLine(QPointF(sx, vis_top), QPointF(sx, vis_bot))
            cy = (vis_top + vis_bot) / 2
            hr = QRectF(sx - 21, cy - 21, 42, 42)
            paint_glass(p, hr, 21, 1.5, None, glow=0.8, tint=0.5)
            draw_icon(p, "arrows", QRectF(sx - 11, cy - 11, 22, 22), cream(255), 1.6)
            # labels
            if sx - max(0.0, dest.left()) > 120:
                self._pill(p, "Original", max(12.0, dest.left() + 12), max(12.0, dest.top() + 12))
            if min(W, dest.right()) - sx > 120:
                self._pill(p, "Enhanced", min(W - 12.0, dest.right() - 12), max(12.0, dest.top() + 12),
                           align_right=True)

        if self.zoom > 1.02 and self.after is not None:
            pct = int(round(dest.width() / float(self.after.w) * 100))
            self._pill(p, f"{pct}%", W / 2 - 24, 12)
        elif self.zoom > 1.02:
            pct = int(round(dest.width() / float(self._dims()[0]) * 100))
            self._pill(p, f"{pct}%", W / 2 - 24, 12)

    def _render_empty(self, p: QPainter, full: QRectF) -> None:
        hv = self._empty_hover.value
        W, H = full.width(), full.height()
        box = QRectF(W / 2 - 250, H / 2 - 150, 500, 300)
        path = QPainterPath()
        path.addRoundedRect(box, 34, 34)
        g = QRadialGradient(box.center(), 340)
        g.setColorAt(0, warm(int(16 + 14 * hv)))
        g.setColorAt(1, warm(0))
        p.fillPath(path, g)
        pen = QPen(warm(int(70 + 70 * hv)), 1.6, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([3.0, 5.0])
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawPath(path)
        # icon
        ic = QRectF(W / 2 - 34, box.top() + 52, 68, 68)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(cream(int(235 + 20 * hv)))
        p.drawPath(_spark_path(ic.center().x() - 5, ic.center().y() + 2, 30))
        p.drawPath(_spark_path(ic.right() - 6, ic.top() + 12, 11))
        p.setFont(ui_font(24, QFont.Weight.DemiBold))
        p.setPen(cream(250))
        p.drawText(QRectF(box.left(), box.top() + 132, box.width(), 36), Qt.AlignmentFlag.AlignCenter,
                   "Drop a photo or video here")
        p.setFont(ui_font(13.5))
        p.setPen(cream(150))
        p.drawText(QRectF(box.left(), box.top() + 172, box.width(), 24), Qt.AlignmentFlag.AlignCenter,
                   "or click anywhere to choose a file")
        p.setFont(ui_font(11.5))
        p.setPen(cream(95))
        p.drawText(QRectF(box.left(), box.bottom() - 46, box.width(), 22), Qt.AlignmentFlag.AlignCenter,
                   "JPG  PNG  WEBP  TIFF  BMP      MP4  MOV  MKV  AVI  WEBM")

    def paintEvent(self, ev) -> None:
        dpr = self.devicePixelRatioF()
        size = QSize(int(self.width() * dpr), int(self.height() * dpr))
        if self._cache is None or self._dirty or self._cache.size() != size:
            pm = QPixmap(size)
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.GlobalColor.transparent)
            cp = QPainter(pm)
            self._render(cp)
            cp.end()
            self._cache = pm
            self._dirty = False
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        W, H = float(self.width()), float(self.height())
        hv = self._hint.value
        if hv > 0.02 and self.has_media():
            p.setOpacity(hv * 0.9)
            txt = "Drag to compare   -   Scroll to zoom   -   Right-drag to move   -   Double-click to fit" \
                if self.after is not None else "Scroll to zoom   -   Double-click to fit"
            self._pill(p, txt, W / 2 - (QFontMetrics(ui_font(12.5, QFont.Weight.DemiBold)).horizontalAdvance(txt) + 26) / 2,
                       H - 44)
            p.setOpacity(1.0)
        if self._busy:
            p.setFont(ui_font(13.5, QFont.Weight.DemiBold))
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(self._busy)
            w = tw + 64
            r = QRectF(W / 2 - w / 2, H / 2 - 24, w, 48)
            paint_glass(p, r, 24, 1.6, None, glow=0.6)
            p.fillPath(self._rr(r, 24), QColor(0, 0, 0, 120))
            arc = QRectF(r.left() + 16, r.center().y() - 9, 18, 18)
            p.setPen(QPen(cream(70), 2.2))
            p.drawEllipse(arc)
            p.setPen(QPen(cream(255), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(arc, int(-self._spin * 16), int(100 * 16))
            p.setPen(cream(255))
            p.drawText(QRectF(r.left() + 44, r.top(), tw + 8, r.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._busy)
        p.end()

    @staticmethod
    def _rr(r: QRectF, rad: float) -> QPainterPath:
        pth = QPainterPath()
        pth.addRoundedRect(r, rad, rad)
        return pth


# ----------------------------------------------------------------------------
#  Dialogs
# ----------------------------------------------------------------------------
class GlassDialog(QDialog):
    MARGIN = 12

    def __init__(self, parent: Optional[QWidget], title: str, min_width: int = 440):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        m = self.MARGIN
        outer = QVBoxLayout(self)
        outer.setContentsMargins(m + 26, m + 22, m + 26, m + 22)
        outer.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(make_label(title, 19, QFont.Weight.DemiBold))
        head.addStretch(1)
        self.close_btn = WindowButton("close", True)
        self.close_btn.clicked.connect(self.reject)
        head.addWidget(self.close_btn)
        outer.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.body)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(10)
        self.buttons.addStretch(1)
        outer.addSpacing(4)
        outer.addLayout(self.buttons)
        self.setMinimumWidth(min_width + 2 * m + 52)
        LightField.get().register(self)

    def reject(self) -> None:
        self.done(-1)

    def add_button(self, text: str, code: int, primary: bool = False) -> GlassButton:
        b = GlassButton(text, primary=primary, height=40)
        b.clicked.connect(lambda: self.done(code))
        self.buttons.addWidget(b)
        return b

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        m = self.MARGIN
        r = QRectF(self.rect()).adjusted(m, m, -m, -m)
        for i in range(7, 0, -1):
            sh = QPainterPath()
            sh.addRoundedRect(r.adjusted(-i, -i + 4, i, i + 4), 28 + i, 28 + i)
            p.fillPath(sh, QColor(0, 0, 0, 20))
        body = QPainterPath()
        body.addRoundedRect(r, 28, 28)
        bg = QLinearGradient(r.topLeft(), r.bottomRight())
        bg.setColorAt(0, _bg_mix(0.055, 252))
        bg.setColorAt(1, _bg_mix(0.008, 252))
        p.fillPath(body, bg)
        paint_glass(p, r, 28, 1.0, LightField.get().local(self))
        p.end()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and ev.position().y() < 70:
            wh = self.windowHandle()
            if wh is not None:
                wh.startSystemMove()
        super().mousePressEvent(ev)


def show_message(parent: Optional[QWidget], title: str, text: str,
                 buttons: Tuple[str, ...] = ("OK",), primary: int = 0) -> int:
    d = GlassDialog(parent, title)
    lab = make_label(text, 14, alpha=215, wrap=True)
    lab.setMinimumWidth(400)
    lab.setMaximumWidth(520)
    d.body.addWidget(lab)
    for i, b in enumerate(buttons):
        d.add_button(b, i, primary=(i == primary))
    d.exec()
    return d.result()


class SettingsDialog(QDialog):
    pass


class ThemeSwatch(QAbstractButton):
    """A small preview tile for one colour theme."""

    def __init__(self, theme: Theme, current: Callable[[], str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.theme = theme
        self._current = current
        self._hover = Tween(self, 0.0, 160)
        self._hover.changed.connect(lambda: self.update())
        self.setFixedSize(100, 60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def enterEvent(self, ev) -> None:
        self._hover.to(1.0)

    def leaveEvent(self, ev) -> None:
        self._hover.to(0.0)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self.theme
        r = QRectF(self.rect()).adjusted(1.6, 1.6, -1.6, -1.6)
        path = QPainterPath()
        path.addRoundedRect(r, 16, 16)
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        f = min(1.0, t.blob)
        g.setColorAt(0.0, QColor(*_mix(t.bg, t.glow, 0.46 * f)))
        g.setColorAt(0.55, QColor(*t.bg))
        g.setColorAt(1.0, QColor(*_mix(t.bg, t.glow, 0.22 * f)))
        p.fillPath(path, g)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(*t.accent))
        p.drawPath(_spark_path(r.left() + 22, r.top() + 20, 8))
        p.setFont(ui_font(11.5, QFont.Weight.DemiBold))
        p.setPen(QColor(*t.text))
        p.drawText(QRectF(r.left() + 11, r.bottom() - 25, r.width() - 20, 18),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, t.name)
        sel = self._current() == t.key
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(cream(255 if sel else int(70 + 90 * self._hover.value)), 2.2 if sel else 1.2))
        p.drawPath(path)
        p.end()


def open_settings(parent: QWidget, settings: Settings, engine: Engine,
                  on_change: Callable[[str], None]) -> None:
    d = GlassDialog(parent, "Settings", 460)
    scr = QScrollArea()
    scr.setWidgetResizable(True)
    scr.setFrameShape(QFrame.Shape.NoFrame)
    scr.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scr.setStyleSheet(scroll_qss())
    scr.viewport().setAutoFillBackground(False)
    inner = QWidget()
    inner.setAutoFillBackground(False)
    bl = QVBoxLayout(inner)
    bl.setContentsMargins(0, 0, 10, 0)
    bl.setSpacing(12)
    scr.setWidget(inner)
    scr_h = QApplication.primaryScreen().availableGeometry().height() if QApplication.primaryScreen() else 800
    scr.setFixedHeight(int(min(620, max(320, scr_h * 0.62))))
    d.body.addWidget(scr)

    def row(title: str, hint: str, widget: QWidget) -> QHBoxLayout:
        h = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addWidget(make_label(title, 14.5, QFont.Weight.DemiBold))
        if hint:
            col.addWidget(make_label(hint, 11.5, alpha=135))
        h.addLayout(col)
        h.addStretch(1)
        h.addWidget(widget)
        return h

    gpus = engine.gpu_list()
    gpu_ok = bool(gpus)
    sw_gpu = GlassSwitch(settings.use_gpu and gpu_ok)
    sw_gpu.setEnabled(gpu_ok)

    def gpu_text() -> str:
        if not gpu_ok:
            return "No compatible graphics card found - using the processor"
        if engine.on_gpu:
            return f"{engine.gpu_name()}  -  {engine.vram_total_gb():.1f} GB"
        return "Off - using the processor (slower)"

    grow = QHBoxLayout()
    gcol = QVBoxLayout()
    gcol.setSpacing(1)
    gcol.addWidget(make_label("Use graphics card", 14.5, QFont.Weight.DemiBold))
    hint_lab = make_label(gpu_text(), 11.5, alpha=135, wrap=True)
    gcol.addWidget(hint_lab)
    grow.addLayout(gcol, 1)
    grow.addWidget(sw_gpu)
    bl.addLayout(grow)
    seg_gpu = None
    if len(gpus) > 1:
        cur = engine.device.index if engine.on_gpu else max(gpus, key=lambda g: g[2])[0]
        seg_gpu = Segmented([(str(g[0]), f"Card {n + 1}", "") for n, g in enumerate(gpus)], str(cur), 40)
        bl.addWidget(seg_gpu)

    bl.addWidget(make_label("VIDEO", 11, QFont.Weight.DemiBold, 120, 1.8))
    seg_q = Segmented([("high", "Best", ""), ("standard", "Standard", ""), ("small", "Smaller", "")],
                      settings.video_quality, 40)
    bl.addWidget(seg_q)
    seg_c = Segmented([("h264", "Works everywhere", ""), ("h265", "Smaller file", "")],
                      settings.video_codec, 40)
    bl.addWidget(seg_c)
    sw_steady = GlassSwitch(settings.steady_video)
    bl.addLayout(row("Steady video", "Reduces flicker between frames", sw_steady))

    bl.addWidget(make_label("PHOTOS", 11, QFont.Weight.DemiBold, 120, 1.8))
    sl = GlassSlider("JPG quality", settings.jpg_quality, 60, 100, "")
    bl.addWidget(sl)

    bl.addWidget(make_label("DOWNLOADS", 11, QFont.Weight.DemiBold, 120, 1.8))
    dlrow = QHBoxLayout()
    dlcol = QVBoxLayout()
    dlcol.setSpacing(1)
    dlcol.addWidget(make_label("Download folder", 14.5, QFont.Weight.DemiBold))
    dl_hint = make_label(settings.output_dir or str(default_download_dir()), 11.5, alpha=135, wrap=True)
    dlcol.addWidget(dl_hint)
    dlrow.addLayout(dlcol, 1)
    b_dl = GlassButton("Change", "folder", height=34)
    dlrow.addWidget(b_dl)
    bl.addLayout(dlrow)

    def pick_dl_folder() -> None:
        start = settings.output_dir or str(default_download_dir())
        dd = QFileDialog.getExistingDirectory(d, "Where should downloads go?", start)
        if dd:
            settings.output_dir = dd
            settings.save()
            dl_hint.setText(dd)
            on_change("folder")

    b_dl.clicked.connect(lambda _=False: pick_dl_folder())

    bl.addWidget(make_label("THEME", 11, QFont.Weight.DemiBold, 120, 1.8))
    swatches: List[ThemeSwatch] = []
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(10)

    def pick_theme(key: str) -> None:
        settings.theme = key
        settings.save()
        on_change("theme")
        scr.setStyleSheet(scroll_qss())
        restyle_tree(d)
        for sw in swatches:
            sw.update()

    for n, th in enumerate(THEMES):
        sw = ThemeSwatch(th, lambda: settings.theme)
        sw.clicked.connect(lambda _=False, k=th.key: pick_theme(k))
        swatches.append(sw)
        grid.addWidget(sw, n // 4, n % 4)
    bl.addLayout(grid)
    sw_amb = GlassSwitch(settings.ambient)
    bl.addLayout(row("Moving background", "Stars, leaves, petals, bubbles and galaxies", sw_amb))

    def apply_ambient(c: bool) -> None:
        settings.ambient = c
        settings.save()
        on_change("ambient")

    sw_amb.toggled.connect(apply_ambient)

    def apply_gpu(c: bool) -> None:
        settings.use_gpu = c
        settings.save()
        on_change("gpu")
        hint_lab.setText(gpu_text())

    sw_gpu.toggled.connect(apply_gpu)
    if seg_gpu is not None:
        def pick_gpu(k: str) -> None:
            settings.gpu_index = int(k)
            settings.save()
            on_change("gpu")
            hint_lab.setText(gpu_text())
        seg_gpu.changed.connect(pick_gpu)
    seg_q.changed.connect(lambda k: (setattr(settings, "video_quality", k), settings.save()))
    seg_c.changed.connect(lambda k: (setattr(settings, "video_codec", k), settings.save()))
    sw_steady.toggled.connect(lambda c: (setattr(settings, "steady_video", c), settings.save()))
    sl.valueChanged.connect(lambda v: (setattr(settings, "jpg_quality", v), settings.save()))

    b1 = GlassButton("Open files folder", "folder", height=40)
    b1.clicked.connect(lambda: open_folder(MODELS_DIR))
    d.buttons.insertWidget(0, b1)
    d.add_button("Done", 1, primary=True)
    d.exec()


# ----------------------------------------------------------------------------
#  Queue strip
# ----------------------------------------------------------------------------
class QueueList(QListWidget):
    removeRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setViewMode(QListWidget.ViewMode.ListMode)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.restyle()
        self.viewport().setAutoFillBackground(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def restyle(self) -> None:
        self.setStyleSheet("QListWidget { background: transparent; border: none; outline: none; }"
                           + scrollbar_qss())

    def wheelEvent(self, ev) -> None:
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - ev.angleDelta().y())
        ev.accept()

    def keyPressEvent(self, ev) -> None:
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.removeRequested.emit()
        else:
            super().keyPressEvent(ev)


class QueueDelegate(QStyledItemDelegate):
    W, H = 138, 104

    def sizeHint(self, option, index) -> QSize:
        return QSize(self.W, self.H)

    def paint(self, p: QPainter, option, index) -> None:
        job: Optional[Job] = index.data(Qt.ItemDataRole.UserRole)
        if job is None:
            return
        S = QtWidgets.QStyle.StateFlag
        selected = bool(option.state & S.State_Selected)
        hover = bool(option.state & S.State_MouseOver)
        r = QRectF(option.rect).adjusted(5, 5, -5, -5)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(r, 15, 15)
        p.save()
        p.setClipPath(path)
        p.fillRect(r, QColor(10, 10, 10))
        if job.thumb is not None and not job.thumb.isNull():
            tw, th = job.thumb.width(), job.thumb.height()
            ar = r.width() / r.height()
            if tw / th > ar:
                sw = th * ar
                src = QRectF((tw - sw) / 2, 0, sw, th)
            else:
                sh = tw / ar
                src = QRectF(0, (th - sh) / 2, tw, sh)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.setOpacity(1.0 if (selected or hover) else 0.82)
            p.drawImage(r, job.thumb, src)
            p.setOpacity(1.0)
        else:
            draw_icon(p, "film" if job.kind == "video" else "image",
                      QRectF(r.center().x() - 16, r.center().y() - 22, 32, 32), cream(130))
        shade = QLinearGradient(QPointF(0, r.bottom() - 40), QPointF(0, r.bottom()))
        shade.setColorAt(0, QColor(0, 0, 0, 0))
        shade.setColorAt(1, QColor(0, 0, 0, 215))
        p.fillRect(QRectF(r.left(), r.bottom() - 40, r.width(), 40), shade)
        if job.status == "working":
            p.fillRect(r, QColor(0, 0, 0, 120))
        p.restore()

        # name
        p.setFont(ui_font(11.5, QFont.Weight.Medium))
        p.setPen(cream(240))
        fm = QFontMetrics(p.font())
        name = fm.elidedText(job.name, Qt.TextElideMode.ElideMiddle, int(r.width() - 16))
        p.drawText(QRectF(r.left() + 8, r.bottom() - 24, r.width() - 16, 18),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)

        # status badge
        badge = QRectF(r.right() - 28, r.top() + 7, 21, 21)
        if job.status == "done":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(cream(235))
            p.drawEllipse(badge)
            draw_icon(p, "check", badge.adjusted(2, 2, -2, -2), QColor(8, 8, 8), 1.8)
        elif job.status == "error":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(190, 70, 60, 235))
            p.drawEllipse(badge)
            draw_icon(p, "warn", badge.adjusted(2, 2, -2, -2), QColor(255, 244, 236), 1.8)
        elif job.status == "working":
            c = r.center()
            rr = QRectF(c.x() - 20, c.y() - 20, 40, 40)
            p.setPen(QPen(cream(60), 3))
            p.drawEllipse(rr)
            p.setPen(QPen(cream(255), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rr, 90 * 16, int(-max(0.03, job.progress) * 360 * 16))
            p.setFont(ui_font(11, QFont.Weight.DemiBold))
            p.setPen(cream(255))
            p.drawText(rr, Qt.AlignmentFlag.AlignCenter, f"{int(job.progress * 100)}")
        if job.kind == "video":
            p.setFont(ui_font(9.5, QFont.Weight.Bold))
            vr = QRectF(r.left() + 7, r.top() + 7, 34, 17)
            vp = QPainterPath()
            vp.addRoundedRect(vr, 8, 8)
            p.fillPath(vp, QColor(0, 0, 0, 170))
            p.setPen(cream(235))
            p.drawText(vr, Qt.AlignmentFlag.AlignCenter, "VIDEO")

        # rim
        rim = QLinearGradient(r.topLeft(), r.bottomRight())
        a = 235 if selected else (130 if hover else 70)
        rim.setColorAt(0, warm(a))
        rim.setColorAt(0.5, warm(int(a * 0.25)))
        rim.setColorAt(1, warm(int(a * 0.7)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QBrush(rim), 2.0 if selected else 1.2))
        p.drawPath(path)
        p.restore()


# ----------------------------------------------------------------------------
#  Small helper widgets for the main window
# ----------------------------------------------------------------------------
class TurboSegmented(Segmented):
    """Quality chooser with the 4th, video-only option.  Its edge softly glows."""

    def __init__(self, options, value: str, height: int = 58, parent: Optional[QWidget] = None):
        super().__init__(options, value, height, parent)
        self._phase = 0.0
        self._pulse = QTimer(self)
        self._pulse.setInterval(45)
        self._pulse.timeout.connect(self._tick)

    def showEvent(self, ev) -> None:
        self._pulse.start()
        super().showEvent(ev)

    def hideEvent(self, ev) -> None:
        self._pulse.stop()
        super().hideEvent(ev)

    def _tick(self) -> None:
        self._phase = (self._phase + 0.07) % (2.0 * math.pi)
        self.update()

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        n = len(self.options)
        r = QRectF(self.rect()).adjusted(0.8, 0.8, -0.8, -0.8)
        seg = (r.width() - 6) / float(n)
        cell = QRectF(r.left() + 3 + seg * (n - 1) + 3, r.top() + 6, seg - 9, r.height() - 12)
        rad = min(cell.height() / 2.0, 20.0)
        pulse = 0.5 + 0.5 * math.sin(self._phase)
        base = (0.55 + 0.45 * pulse) * (1.5 if self.value() == self.keys[-1] else 1.0)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(cell, rad, rad)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(5, 0, -1):                         # soft outer glow
            p.setPen(QPen(_accent((26 - i * 4) * base), 1.0 + i * 1.5))
            p.drawPath(path)
        grad = QLinearGradient(cell.topLeft(), cell.bottomRight())
        grad.setColorAt(0.0, _accent(150 + 80 * pulse))
        grad.setColorAt(0.5, warm(60))
        grad.setColorAt(1.0, _accent(150 + 80 * (1.0 - pulse)))
        p.setPen(QPen(QBrush(grad), 1.4))
        p.drawPath(path)
        p.end()


class ElideLabel(QLabel):
    def __init__(self, text: str = "", px: float = 12.5, alpha: int = 170, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = text
        self._alpha = alpha
        self.setFont(ui_font(px))
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setFullText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self.update()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setFont(self.font())
        p.setPen(cream(self._alpha))
        el = QFontMetrics(self.font()).elidedText(self._full, Qt.TextElideMode.ElideMiddle, self.width())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, el)
        p.end()


class TitleBar(QWidget):
    def __init__(self, win: "YokoWindow"):
        super().__init__(win)
        self.win = win
        self.setFixedHeight(54)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 4, 12, 0)
        lay.setSpacing(2)
        lay.addSpacing(210)
        lay.addStretch(1)
        self.chip = make_label("", 12, QFont.Weight.Medium, 150)
        lay.addWidget(self.chip)
        lay.addSpacing(10)
        self.btn_settings = WindowButton("gear")
        self.btn_min = WindowButton("min")
        self.btn_max = WindowButton("max")
        self.btn_close = WindowButton("close", True)
        for b in (self.btn_settings, self.btn_min, self.btn_max, self.btn_close):
            lay.addWidget(b)
        self.btn_settings.clicked.connect(win.open_settings_dialog)
        self.btn_min.clicked.connect(win.showMinimized)
        self.btn_max.clicked.connect(win.toggle_max)
        self.btn_close.clicked.connect(win.close)

    def set_chip(self, text: str) -> None:
        self.chip.setText(text)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        logo = QRectF(20, 11, 34, 34)
        paint_glass(p, logo, 11, 1.3, LightField.get().local(self), glow=0.5, tint=0.3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(cream(250))
        p.drawPath(_spark_path(logo.center().x() - 1.5, logo.center().y() + 1, 11))
        p.drawPath(_spark_path(logo.right() - 8, logo.top() + 9, 4))
        p.setFont(ui_font(19, QFont.Weight.Bold, 0.2))
        p.setPen(cream(255))
        p.drawText(QRectF(64, 0, 80, 54), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "Yoko")
        w = QFontMetrics(p.font()).horizontalAdvance("Yoko")
        p.setFont(ui_font(19, QFont.Weight.Light, 0.2))
        p.setPen(cream(190))
        p.drawText(QRectF(64 + w + 4, 0, 120, 54), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   "Enhancer")
        p.end()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            wh = self.win.windowHandle()
            if wh is not None:
                wh.startSystemMove()

    def mouseDoubleClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.win.toggle_max()


# ============================================================================
#                              M A I N   W I N D O W
# ============================================================================
# ----------------------------------------------------------------------------
#  Living background  (stars / autumn leaves / sakura / underwater / galaxy)
#  Drawn behind the glass, one scene per colour theme
# ----------------------------------------------------------------------------
import random as _random
from PySide6.QtGui import QPolygonF, QTransform


def _make_leaf_sprite(color: Tuple[int, int, int], shape: int, size: int = 64) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = size / 64.0
    p.scale(s, s)
    path = QPainterPath()
    if shape == 0:      # simple leaf
        path.moveTo(32, 5)
        path.cubicTo(60, 18, 58, 46, 32, 58)
        path.cubicTo(6, 46, 4, 18, 32, 5)
    else:               # maple-like leaf
        half = [(32, 4), (36, 15), (46, 10), (44, 24), (57, 23), (48, 33), (54, 43),
                (39, 42), (34, 52), (32, 58)]
        path.moveTo(*half[0])
        for pt in half[1:]:
            path.lineTo(*pt)
        for pt in reversed(half[:-1]):
            path.lineTo(64 - pt[0], pt[1])
        path.closeSubpath()
    light = QColor(*_mix(color, (255, 240, 215), 0.30))
    dark = QColor(*_mix(color, (60, 25, 15), 0.30))
    g = QLinearGradient(QPointF(10, 6), QPointF(54, 58))
    g.setColorAt(0.0, light)
    g.setColorAt(1.0, dark)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawPath(path)
    vein = QColor(*_mix(color, (60, 25, 15), 0.55))
    vein.setAlpha(150)
    p.setPen(QPen(vein, 1.3))
    p.drawLine(QPointF(32, 10), QPointF(32, 60))
    if shape == 0:
        for yy in (22, 32, 42):
            p.drawLine(QPointF(32, yy), QPointF(32 + 11, yy - 8))
            p.drawLine(QPointF(32, yy), QPointF(32 - 11, yy - 8))
    p.end()
    return pm


def _make_petal_sprite(color: Tuple[int, int, int], size: int = 64) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = size / 64.0
    p.scale(s, s)
    path = QPainterPath()
    path.moveTo(32, 60)
    path.cubicTo(14, 48, 5, 28, 14, 13)
    path.cubicTo(19, 5, 29, 6, 32, 15)
    path.cubicTo(35, 6, 45, 5, 50, 13)
    path.cubicTo(59, 28, 50, 48, 32, 60)
    g = QRadialGradient(QPointF(32, 56), 52)
    g.setColorAt(0.0, QColor(*_mix(color, (240, 120, 160), 0.55)))
    g.setColorAt(0.45, QColor(*color))
    g.setColorAt(1.0, QColor(*_mix(color, (255, 255, 255), 0.55)))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawPath(path)
    p.end()
    return pm


def _make_glow_sprite(color: Tuple[int, int, int], size: int = 48) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    g = QRadialGradient(QPointF(size / 2, size / 2), size / 2)
    g.setColorAt(0.0, QColor(color[0], color[1], color[2], 255))
    g.setColorAt(0.18, QColor(color[0], color[1], color[2], 150))
    g.setColorAt(1.0, QColor(color[0], color[1], color[2], 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(g))
    p.drawEllipse(QRectF(0, 0, size, size))
    p.end()
    return pm


def _make_galaxy_sprite(size: int = 520, seed: int = 7) -> QPixmap:
    rnd = _random.Random(seed)
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    c = size / 2.0
    halo = QRadialGradient(QPointF(c, c), c)
    halo.setColorAt(0.0, QColor(214, 184, 255, 120))
    halo.setColorAt(0.35, QColor(168, 130, 245, 52))
    halo.setColorAt(1.0, QColor(120, 80, 220, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(halo))
    p.drawEllipse(QRectF(0, 0, size, size))
    cols = [(236, 222, 255), (255, 206, 236), (190, 206, 255), (255, 232, 246)]
    turns = 3.3 * math.pi
    for arm in range(2):
        off = arm * math.pi
        for _ in range(900):
            f = rnd.random() ** 0.85
            th = f * turns
            r = 10 + f * (c - 24)
            th += rnd.gauss(0, 0.16)
            r += rnd.gauss(0, 3 + r * 0.07)
            x = c + math.cos(th + off) * r
            y = c + math.sin(th + off) * r
            col = cols[rnd.randrange(len(cols))]
            a = int(clamp(rnd.uniform(60, 215) * (1.0 - f * 0.55), 20, 230))
            rad = rnd.uniform(0.5, 1.7)
            p.setBrush(QColor(col[0], col[1], col[2], a))
            p.drawEllipse(QPointF(x, y), rad, rad)
    core = QRadialGradient(QPointF(c, c), 46)
    core.setColorAt(0.0, QColor(255, 246, 255, 235))
    core.setColorAt(0.3, QColor(240, 214, 255, 150))
    core.setColorAt(1.0, QColor(190, 150, 255, 0))
    p.setBrush(QBrush(core))
    p.drawEllipse(QRectF(c - 46, c - 46, 92, 92))
    p.end()
    return pm


class AmbientScene:
    """Small drifting objects behind the glass.  Everything is cheap QPainter work."""

    def __init__(self) -> None:
        self.key = ""
        self.W = 0.0
        self.H = 0.0
        self.t = 0.0
        self.last = time.time()
        self.rnd = _random.Random()
        self.items: List[dict] = []
        self.novae: List[dict] = []
        self.rest: List[dict] = []      # leaves / petals lying on the bottom
        self.next_nova = 3.0
        self.sprites: Dict[str, List[QPixmap]] = {}
        self.glow: Dict[Tuple[int, int, int], QPixmap] = {}
        self.galaxy: Optional[QPixmap] = None
        self.kind = "none"

    # ------------------------------------------------------------------ setup
    @staticmethod
    def kind_for(key: str) -> str:
        return {"ivory": "stars", "obsidian": "stars", "sunset": "leaves", "rose": "petals",
                "ocean": "water", "amethyst": "galaxy", "emerald": "leaves"}.get(key, "dust")

    def _glow_sprite(self, col: Tuple[int, int, int]) -> QPixmap:
        if col not in self.glow:
            self.glow[col] = _make_glow_sprite(col)
        return self.glow[col]

    def reset(self, key: str, W: float, H: float) -> None:
        self.key = key
        self.W, self.H = W, H
        self.kind = self.kind_for(key)
        self.items = []
        self.novae = []
        self.rest = []
        self.next_nova = self.t + 2.0
        r = self.rnd
        area = W * H
        if self.kind in ("stars", "dust", "galaxy"):
            n = int(area / (13000 if self.kind != "dust" else 30000))
            for _ in range(n):
                big = r.random() < 0.10
                self.items.append(dict(
                    x=r.uniform(0, W), y=r.uniform(0, H),
                    r=r.uniform(1.3, 2.1) if big else r.uniform(0.55, 1.25),
                    ph=r.uniform(0, 6.28), sp=r.uniform(0.5, 1.9), big=big,
                    vx=r.uniform(-1.5, 1.5),
                    vy=(-r.uniform(1.5, 6.0) if key == "ivory" else r.uniform(-0.8, 0.8))))
        elif self.kind == "leaves":
            if key == "emerald":
                pal = [(118, 196, 150), (96, 176, 132), (150, 214, 168), (84, 150, 116),
                       (172, 212, 150), (108, 170, 140)]
            else:
                pal = [(214, 118, 66), (192, 92, 58), (226, 154, 80), (164, 86, 60), (190, 142, 78), (148, 104, 66)]
            self.sprites["leaves"] = [_make_leaf_sprite(c, s) for c in pal for s in (0, 1)]
            for _ in range(int(area / 32000)):
                self.items.append(self._new_leaf(True))
            for _ in range(9):
                self._land(self._new_leaf(True), True)
        elif self.kind == "petals":
            pal = [(255, 216, 228), (252, 198, 216), (255, 230, 238), (246, 184, 206), (255, 208, 222)]
            self.sprites["petals"] = [_make_petal_sprite(c) for c in pal]
            for _ in range(int(area / 30000)):
                self.items.append(self._new_petal(True))
            for _ in range(16):
                self._land(self._new_petal(True), True)
        elif self.kind == "water":
            for _ in range(int(area / 28000)):
                self.items.append(self._new_bubble(True))
            for _ in range(int(area / 40000)):
                self.items.append(dict(speck=True, x=r.uniform(0, W), y=r.uniform(0, H),
                                       r=r.uniform(0.6, 1.3), ph=r.uniform(0, 6.28),
                                       vx=r.uniform(-4, 4), vy=r.uniform(-6, 2)))
        if self.kind == "galaxy" and self.galaxy is None:
            self.galaxy = _make_galaxy_sprite()

    def _new_leaf(self, anywhere: bool) -> dict:
        r = self.rnd
        sprites = self.sprites["leaves"]
        return dict(x=r.uniform(-20, self.W + 20), y=r.uniform(-self.H, self.H) if anywhere else r.uniform(-90, -20),
                    sz=r.uniform(15, 28), vy=r.uniform(20, 44), amp=r.uniform(18, 52),
                    fr=r.uniform(0.35, 0.95), ph=r.uniform(0, 6.28), rot=r.uniform(0, 360),
                    rv=r.uniform(-55, 55), tp=r.uniform(0, 6.28), tv=r.uniform(0.8, 2.0),
                    sp=r.randrange(len(sprites)), a=r.uniform(0.55, 0.85), bx=0.0)

    def _new_petal(self, anywhere: bool) -> dict:
        r = self.rnd
        n = len(self.sprites["petals"])
        return dict(x=r.uniform(-20, self.W + 20), y=r.uniform(-self.H, self.H) if anywhere else r.uniform(-80, -16),
                    sz=r.uniform(11, 20), vy=r.uniform(24, 48), amp=r.uniform(20, 60),
                    fr=r.uniform(0.4, 1.1), ph=r.uniform(0, 6.28), rot=r.uniform(0, 360),
                    rv=r.uniform(-70, 70), tp=r.uniform(0, 6.28), tv=r.uniform(1.0, 2.4),
                    sp=r.randrange(n), a=r.uniform(0.6, 0.95), stuck=False, life=0.0, age=0.0, bx=0.0)

    def _land(self, it: dict, seed: bool = False) -> None:
        """A falling leaf / petal comes to rest on the bottom edge for a while."""
        r = self.rnd
        x = r.uniform(18, self.W - 18) if seed else clamp(it["x"] + it["bx"], 14, self.W - 14)
        life = r.uniform(28, 60)
        self.rest.append(dict(x=x, y=self.H - r.uniform(3, 22), rot=r.uniform(0, 360), sz=it["sz"],
                              sp=it["sp"], a=it["a"], life=life,
                              age=r.uniform(0, life * 0.6) if seed else 0.0))

    def _maybe_land(self, it: dict, kind: str) -> None:
        cap = 40 if kind == "petals" else 18
        n = len(self.rest)
        chance = 0.75 if n < cap * 0.5 else (0.4 if n < cap else 0.0)
        if self.rnd.random() < chance:
            self._land(it)

    def _new_bubble(self, anywhere: bool) -> dict:
        r = self.rnd
        return dict(bubble=True, x=r.uniform(0, self.W), y=r.uniform(0, self.H) if anywhere else self.H + r.uniform(5, 60),
                    r=r.uniform(2.0, 7.0), vy=r.uniform(18, 52), amp=r.uniform(4, 14),
                    fr=r.uniform(0.6, 1.6), ph=r.uniform(0, 6.28), a=r.uniform(0.45, 0.9), bx=0.0)

    # ------------------------------------------------------------------- step
    def step(self, key: str, W: float, H: float) -> None:
        now = time.time()
        dt = min(0.1, max(0.0, now - self.last))
        self.last = now
        self.t += dt
        if key != self.key or self.W <= 0:
            self.reset(key, W, H)
        elif abs(W - self.W) > 1 or abs(H - self.H) > 1:
            fx, fy = W / self.W, H / self.H
            for it in self.items + self.rest:
                it["x"] *= fx
                it["y"] *= fy
            self.W, self.H = W, H
        r = self.rnd
        k = self.kind
        t = self.t
        for it in self.items:
            if k in ("stars", "dust", "galaxy") or it.get("speck"):
                it["x"] += it["vx"] * dt
                it["y"] += it["vy"] * dt
                if it["x"] < -4:
                    it["x"] = W + 4
                elif it["x"] > W + 4:
                    it["x"] = -4
                if it["y"] < -4:
                    it["y"] = H + 4
                elif it["y"] > H + 4:
                    it["y"] = -4
            elif k in ("leaves", "petals"):
                it["y"] += it["vy"] * dt
                it["bx"] = math.sin(t * it["fr"] + it["ph"]) * it["amp"]
                it["x"] += (7.0 if k == "leaves" else 9.0) * dt
                it["rot"] += it["rv"] * dt
                it["tp"] += it["tv"] * dt
                if it["y"] > H - 16:
                    self._maybe_land(it, k)
                    it.update(self._new_leaf(False) if k == "leaves" else self._new_petal(False))
                elif it["x"] > W + 60:
                    if k == "leaves":
                        it.update(self._new_leaf(False))
                    else:
                        it["x"] = -30
            elif k == "water" and it.get("bubble"):
                it["y"] -= it["vy"] * dt
                it["bx"] = math.sin(t * it["fr"] + it["ph"]) * it["amp"]
                if it["y"] < -12:
                    it.update(self._new_bubble(False))
        if self.rest:
            for o in self.rest:
                o["age"] += dt
            self.rest = [o for o in self.rest if o["age"] < o["life"]]
        if k == "galaxy":
            self.novae = [n for n in self.novae if t - n["t0"] < n["dur"]]
            if t >= self.next_nova and len(self.novae) < 2:
                self.next_nova = t + r.uniform(4.5, 9.5)
                self.novae.append(dict(
                    x=r.uniform(0.08, 0.92) * W, y=r.uniform(0.08, 0.92) * H, t0=t,
                    dur=r.uniform(3.0, 4.2), mr=r.uniform(55, 105),
                    col=r.choice([(255, 214, 240), (214, 200, 255), (200, 226, 255)]),
                    deb=[(r.uniform(0, 6.28), r.uniform(0.5, 1.0)) for _ in range(12)]))

    # ------------------------------------------------------------------ paint
    def paint(self, p: QPainter, W: float, H: float) -> None:
        self.step(CUR.theme.key, W, H)
        k = self.kind
        t = self.t
        p.save()
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            if k in ("stars", "dust", "galaxy"):
                self._paint_stars(p)
            if k == "galaxy":
                self._paint_galaxy(p, W, H)
            elif k == "leaves":
                self._paint_sprites(p, "leaves", 0.9)
                self._paint_rest(p, "leaves", 0.9, 0.5)
            elif k == "petals":
                self._paint_sprites(p, "petals", 1.0)
                self._paint_rest(p, "petals", 1.0, 0.55)
            elif k == "water":
                self._paint_water(p, W, H)
        finally:
            p.restore()

    def _paint_stars(self, p: QPainter) -> None:
        th = CUR.theme
        key = th.key
        if key == "ivory":
            col = (255, 244, 222)
        elif key == "obsidian":
            col = (255, 255, 255)
        elif key == "amethyst":
            col = (236, 226, 255)
        else:
            col = _mix(th.rim, (255, 255, 255), 0.35)
        t = self.t
        glow = self._glow_sprite(col)
        for it in self.items:
            if it.get("speck"):
                continue
            tw = 0.5 + 0.5 * math.sin(t * it["sp"] + it["ph"])
            a = 0.30 + 0.70 * tw
            x, y, rad = it["x"], it["y"], it["r"]
            if it["big"]:
                gs = rad * 9
                p.setOpacity(0.45 * a)
                p.drawPixmap(QRectF(x - gs, y - gs, gs * 2, gs * 2), glow, QRectF(glow.rect()))
                p.setOpacity(1.0)
                p.setPen(QPen(QColor(col[0], col[1], col[2], int(120 * a)), 0.8))
                L = rad * 4.2 * (0.6 + 0.4 * tw)
                p.drawLine(QPointF(x - L, y), QPointF(x + L, y))
                p.drawLine(QPointF(x, y - L), QPointF(x, y + L))
                p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(col[0], col[1], col[2], int(235 * a)))
            p.drawEllipse(QPointF(x, y), rad, rad)

    def _paint_sprites(self, p: QPainter, name: str, base_op: float) -> None:
        sprites = self.sprites[name]
        for it in self.items:
            x = it["x"] + it["bx"]
            y = it["y"]
            if y < -40 or y > self.H + 40:
                continue
            sx = abs(math.cos(it["tp"])) * 0.85 + 0.15
            p.save()
            p.translate(x, y)
            p.rotate(it["rot"])
            p.scale(sx, 1.0)
            p.setOpacity(it["a"] * base_op)
            sz = it["sz"]
            pm = sprites[it["sp"]]
            p.drawPixmap(QRectF(-sz / 2, -sz / 2, sz, sz), pm, QRectF(pm.rect()))
            p.restore()
        p.setOpacity(1.0)

    def _paint_rest(self, p: QPainter, name: str, base_op: float, flat: float) -> None:
        sprites = self.sprites[name]
        for o in self.rest:
            life, age = o["life"], o["age"]
            fade = clamp(min(age / 1.5, (life - age) / 4.0), 0.0, 1.0)
            if fade <= 0:
                continue
            p.save()
            p.translate(o["x"], o["y"])
            p.rotate(o["rot"])
            p.scale(1.0, flat)
            p.setOpacity(o["a"] * base_op * fade)
            sz = o["sz"]
            pm = sprites[o["sp"]]
            p.drawPixmap(QRectF(-sz / 2, -sz / 2, sz, sz), pm, QRectF(pm.rect()))
            p.restore()
        p.setOpacity(1.0)

    def _paint_galaxy(self, p: QPainter, W: float, H: float) -> None:
        t = self.t
        gal = self.galaxy
        if gal is not None:
            for (fx, fy, sc, spd, tilt, op) in ((0.74, 0.30, 1.0, 1.6, -18, 0.80),
                                               (0.16, 0.80, 0.50, -2.4, 24, 0.60)):
                p.save()
                p.translate(fx * W, fy * H)
                p.rotate(tilt)
                p.scale(sc, sc * 0.46)
                p.rotate(t * spd)
                p.setOpacity(op)
                s = float(gal.width())
                p.drawPixmap(QRectF(-s / 2, -s / 2, s, s), gal, QRectF(gal.rect()))
                p.restore()
            p.setOpacity(1.0)
        for n in self.novae:
            age = self.t - n["t0"]
            ph = clamp(age / n["dur"], 0.0, 1.0)
            ramp = min(1.0, ph / 0.07)
            inten = ramp * math.exp(-ph * 3.2)
            col = n["col"]
            x, y, mr = n["x"], n["y"], n["mr"]
            cr = 8 + 34 * inten + 6 * ph
            g = QRadialGradient(QPointF(x, y), cr)
            g.setColorAt(0.0, QColor(255, 255, 255, int(255 * inten)))
            g.setColorAt(0.25, QColor(col[0], col[1], col[2], int(190 * inten)))
            g.setColorAt(1.0, QColor(col[0], col[1], col[2], 0))
            p.setBrush(QBrush(g))
            p.drawEllipse(QPointF(x, y), cr, cr)
            L = mr * 0.95 * inten
            if L > 2:
                p.setPen(QPen(QColor(255, 246, 255, int(170 * inten)), 1.1))
                p.drawLine(QPointF(x - L, y), QPointF(x + L, y))
                p.drawLine(QPointF(x, y - L * 0.8), QPointF(x, y + L * 0.8))
                p.setPen(QPen(QColor(col[0], col[1], col[2], int(90 * inten)), 0.9))
                d = L * 0.55
                p.drawLine(QPointF(x - d, y - d), QPointF(x + d, y + d))
                p.drawLine(QPointF(x - d, y + d), QPointF(x + d, y - d))
            rr = mr * (1.0 - (1.0 - ph) ** 2)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(col[0], col[1], col[2], int(150 * (1 - ph))), 0.6 + 1.6 * (1 - ph)))
            p.drawEllipse(QPointF(x, y), rr, rr)
            p.setPen(Qt.PenStyle.NoPen)
            for ang, spd in n["deb"]:
                dd = rr * spd * 1.15
                a = int(200 * (1 - ph) ** 1.5)
                p.setBrush(QColor(255, 240, 250, a))
                p.drawEllipse(QPointF(x + math.cos(ang) * dd, y + math.sin(ang) * dd), 1.2, 1.2)

    def _paint_water(self, p: QPainter, W: float, H: float) -> None:
        t = self.t
        # soft rays of light from the surface
        for i in range(5):
            cx = W * (0.08 + 0.2 * i) + math.sin(t * 0.17 + i * 1.9) * 70
            sw = 40 + 26 * math.sin(t * 0.23 + i)
            lean = 150 + 60 * math.sin(t * 0.13 + i * 0.7)
            poly = QPolygonF([QPointF(cx - sw * 0.4, -10), QPointF(cx + sw * 0.4, -10),
                              QPointF(cx + lean + sw, H * 0.85), QPointF(cx + lean - sw, H * 0.85)])
            g = QLinearGradient(QPointF(cx, 0), QPointF(cx + lean, H * 0.85))
            a = 26 + 12 * math.sin(t * 0.4 + i * 2.3)
            g.setColorAt(0.0, QColor(196, 232, 252, int(a)))
            g.setColorAt(1.0, QColor(196, 232, 252, 0))
            p.setBrush(QBrush(g))
            p.drawPolygon(poly)
        # specks and bubbles
        for it in self.items:
            if it.get("speck"):
                a = 0.35 + 0.35 * math.sin(t * 0.9 + it["ph"])
                p.setBrush(QColor(206, 238, 252, int(150 * a)))
                p.drawEllipse(QPointF(it["x"], it["y"]), it["r"], it["r"])
            elif it.get("bubble"):
                x = it["x"] + it["bx"]
                y = it["y"]
                rad = it["r"]
                a = it["a"]
                p.setBrush(QColor(210, 240, 255, int(20 * a)))
                p.setPen(QPen(QColor(214, 242, 255, int(120 * a)), 1.0))
                p.drawEllipse(QPointF(x, y), rad, rad)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, int(150 * a)))
                p.drawEllipse(QPointF(x - rad * 0.35, y - rad * 0.35), rad * 0.22, rad * 0.22)
        # jiggly waves at the bottom
        layers = [(H - 78, 11, 0.012, 0.9, (150, 206, 240, 26)),
                  (H - 50, 9, 0.017, -1.2, (170, 218, 246, 30)),
                  (H - 24, 7, 0.023, 1.5, (190, 230, 250, 36))]
        step = 10
        for li, (base, amp, k1, w1, rgba) in enumerate(layers):
            path = QPainterPath()
            path.moveTo(-step, H + 4)
            x = -step
            while x <= W + step:
                y = (base
                     + amp * math.sin(x * k1 + t * w1 + li)
                     + amp * 0.55 * math.sin(x * k1 * 2.3 - t * w1 * 1.4 + li * 2.1)
                     + 1.6 * math.sin(x * 0.09 + t * 3.2 + li * 1.3))      # the little jiggle
                path.lineTo(x, y)
                x += step
            path.lineTo(W + step, H + 4)
            path.closeSubpath()
            gg = QLinearGradient(QPointF(0, base - amp), QPointF(0, H))
            gg.setColorAt(0.0, QColor(rgba[0], rgba[1], rgba[2], rgba[3]))
            gg.setColorAt(1.0, QColor(rgba[0], rgba[1], rgba[2], rgba[3] // 3))
            p.setBrush(QBrush(gg))
            p.drawPath(path)


class YokoWindow(QWidget):
    BORDER = 6
    FILE_FILTER = ("Photos and videos (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXTS | VIDEO_EXTS)) + ");;"
                   "All files (*)")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_app_icon())
        self.resize(1400, 880)
        self.setMinimumSize(1160, 740)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.settings = Settings.load()
        if self.settings.quality == "turbo":                # Turbo is video-only: start from the photo level
            self.settings.quality = self.settings.photo_quality if self.settings.photo_quality in ("maximum", "balanced", "fast") else "maximum"
        set_theme(self.settings.theme)
        self.engine = Engine(self.settings.use_gpu, self.settings.gpu_index)
        self.jobs: List[Job] = []
        self.current = -1
        self.worker: Optional[EnhanceWorker] = None
        self.preview_worker: Optional[PreviewWorker] = None
        self._preview_rect: Optional[Tuple[int, int, int, int, int, int]] = None
        self._cur_size = (0, 0)
        self._selecting = False
        self._errors: List[str] = []
        self._done_count = 0
        self._last_out = ""
        self._result_cache: Dict[int, "np.ndarray"] = {}

        self._build_ui()
        self._wire_shortcuts()
        self._refresh_chip()

        self._ambient = AmbientScene()
        self._bg = QTimer(self)
        self._bg.setInterval(40 if self.settings.ambient else 160)
        self._bg.timeout.connect(self.update)
        self._bg.start()

        if self.settings.use_gpu and not torch.cuda.is_available():
            QTimer.singleShot(700, self._warn_no_gpu)

    # ------------------------------------------------------------------ build
    def _heading(self, text: str) -> QLabel:
        lab = make_label(text, 11, QFont.Weight.DemiBold, 125, 1.8)
        lab.setContentsMargins(4, 0, 0, 0)
        return lab

    def _build_ui(self) -> None:
        B = self.BORDER
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(B, B, B, B)
        self.root.setSpacing(0)
        self.title = TitleBar(self)
        self.root.addWidget(self.title)

        body = QHBoxLayout()
        body.setContentsMargins(18, 4, 18, 18)
        body.setSpacing(18)
        self.root.addLayout(body, 1)

        # ---- left : viewer + queue
        left = QVBoxLayout()
        left.setSpacing(18)
        vp = GlassPanel(28, 0.9)
        vl = QVBoxLayout(vp)
        vl.setContentsMargins(10, 10, 10, 10)
        self.view = CompareView()
        self.view.openRequested.connect(self.open_dialog)
        vl.addWidget(self.view)
        left.addWidget(vp, 1)

        qp = GlassPanel(26, 0.85)
        qp.setFixedHeight(134)
        ql = QHBoxLayout(qp)
        ql.setContentsMargins(12, 8, 14, 8)
        ql.setSpacing(12)
        self.queue = QueueList()
        self.queue.setItemDelegate(QueueDelegate(self.queue))
        self.queue.currentRowChanged.connect(self._on_row_changed)
        self.queue.removeRequested.connect(self.remove_current)
        self.queue.customContextMenuRequested.connect(self._queue_menu)
        ql.addWidget(self.queue, 1)
        qb = QVBoxLayout()
        qb.setSpacing(10)
        qb.addStretch(1)
        self.btn_add = GlassButton("Add", "plus", height=40)
        self.btn_add.setFixedWidth(104)
        self.btn_add.clicked.connect(self.open_dialog)
        self.btn_clear = GlassButton("Clear", "trash", height=40)
        self.btn_clear.setFixedWidth(104)
        self.btn_clear.clicked.connect(self.clear_queue)
        qb.addWidget(self.btn_add)
        qb.addWidget(self.btn_clear)
        qb.addStretch(1)
        ql.addLayout(qb)
        left.addWidget(qp)
        body.addLayout(left, 1)

        # ---- right : controls
        cp = GlassPanel(28, 1.0)
        cp.setFixedWidth(392)
        cl = QVBoxLayout(cp)
        cl.setContentsMargins(20, 20, 20, 20)
        cl.setSpacing(12)

        scroll = self.scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(scroll_qss())
        scroll.viewport().setAutoFillBackground(False)
        content = QWidget()
        content.setAutoFillBackground(False)
        sl = QVBoxLayout(content)
        sl.setContentsMargins(0, 0, 8, 0)
        sl.setSpacing(10)

        s = self.settings
        sl.addWidget(self._heading("QUALITY"))
        self.seg_quality = Segmented([("maximum", "Maximum", "Best detail"),
                                      ("balanced", "Balanced", "Natural"),
                                      ("fast", "Fast", "Quick")], s.quality, 58)
        self.seg_quality.changed.connect(lambda k: self._on_quality(k))
        sl.addWidget(self.seg_quality)
        self.seg_quality4 = TurboSegmented([("maximum", "Maximum", "Best detail"),
                                            ("balanced", "Balanced", "Natural"),
                                            ("fast", "Fast", "Quick"),
                                            ("turbo", "Turbo", "Video boost")],
                                           s.quality if s.quality in ("maximum", "balanced", "fast", "turbo")
                                           else "maximum", 58)
        self.seg_quality4.changed.connect(lambda k: self._on_quality(k))
        self.seg_quality4.hide()
        sl.addWidget(self.seg_quality4)

        sl.addSpacing(6)
        sl.addWidget(self._heading("SIZE"))
        self.seg_size = Segmented([("4k", "4K", ""), ("8k", "8K", ""), ("2x", "2\u00d7", ""),
                                   ("4x", "4\u00d7", ""), ("keep", "Keep", "")], s.target, 42)
        self.seg_size.changed.connect(lambda k: (self._set("target", k), self._update_size_info()))
        sl.addWidget(self.seg_size)
        self.size_info = make_label("", 12, alpha=140)
        self.size_info.setContentsMargins(6, 0, 0, 0)
        sl.addWidget(self.size_info)

        sl.addSpacing(8)
        sl.addWidget(self._heading("ENHANCE"))
        self.opt_blur = ToggleSlider("Remove blur", "Brings back crisp edges", s.remove_blur, s.blur_strength)
        self.opt_blur.changed.connect(lambda c, v: (self._set("remove_blur", c), self._set("blur_strength", v)))
        sl.addWidget(self.opt_blur)
        self.opt_grain = ToggleSlider("Remove grain", "Cleans noise and speckles", s.remove_grain, s.grain_strength)
        self.opt_grain.changed.connect(lambda c, v: (self._set("remove_grain", c), self._set("grain_strength", v)))
        sl.addWidget(self.opt_grain)
        sl.addSpacing(2)
        self.sl_detail = GlassSlider("Detail", s.detail, 0, 100, "%")
        self.sl_detail.valueChanged.connect(lambda v: self._set("detail", v))
        sl.addWidget(self.sl_detail)
        self.sl_texture = GlassSlider("Natural texture", s.texture, 0, 100, "%")
        self.sl_texture.valueChanged.connect(lambda v: self._set("texture", v))
        sl.addWidget(self.sl_texture)
        hint = make_label("Keeps skin and surfaces looking real, never plastic.", 11.5, alpha=125, wrap=True)
        hint.setContentsMargins(8, 0, 0, 0)
        sl.addWidget(hint)

        sl.addSpacing(8)
        sl.addWidget(self._heading("SAVE"))
        self.seg_format = Segmented([("png", "PNG", ""), ("jpg", "JPG", ""), ("webp", "WEBP", ""),
                                     ("tiff", "TIFF", "")], s.output_format, 40)
        self.seg_format.changed.connect(lambda k: self._set("output_format", k))
        sl.addWidget(self.seg_format)
        vf = s.video_format if s.video_format in ("mp4", "mov", "mkv") else "mp4"
        self.seg_vformat = Segmented([("mp4", "MP4", ""), ("mov", "MOV", ""), ("mkv", "MKV", "")], vf, 40)
        self.seg_vformat.changed.connect(lambda k: self._set("video_format", k))
        self.seg_vformat.hide()
        sl.addWidget(self.seg_vformat)
        frow = QHBoxLayout()
        frow.setContentsMargins(6, 2, 0, 0)
        self.out_label = ElideLabel("", 12.5, 170)
        frow.addWidget(self.out_label, 1)
        self.btn_folder = GlassButton("Change", "folder", height=34)
        self.btn_folder.clicked.connect(self.choose_output_dir)
        frow.addWidget(self.btn_folder)
        sl.addLayout(frow)
        self._refresh_out_label()
        sl.addStretch(1)
        scroll.setWidget(content)
        cl.addWidget(scroll, 1)

        # ---- action area
        self.progress = GlassProgress()
        cl.addWidget(self.progress)
        self.status = make_label("Ready", 12.5, alpha=175, wrap=True)
        self.status.setMinimumHeight(34)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        cl.addWidget(self.status)
        self.btn_download = GlassButton("Download", "download", height=40)
        self.btn_download.clicked.connect(lambda: self.on_download())
        self.btn_download.setVisible(False)
        cl.addWidget(self.btn_download)
        self.btn_show = GlassButton("Show in folder", "folder", height=36)
        self.btn_show.clicked.connect(lambda: self._show_saved())
        self.btn_show.setVisible(False)
        cl.addWidget(self.btn_show)
        arow = QHBoxLayout()
        arow.setSpacing(10)
        self.btn_preview = GlassButton("Preview", "eye", height=54)
        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_go = GlassButton("Enhance", "spark", primary=True, height=54)
        self.btn_go.clicked.connect(self.on_enhance)
        arow.addWidget(self.btn_preview, 5)
        arow.addWidget(self.btn_go, 7)
        cl.addLayout(arow)
        body.addWidget(cp)

    def _wire_shortcuts(self) -> None:
        sc = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        sc.activated.connect(self.open_dialog)
        sc2 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        sc2.activated.connect(self.on_enhance)

    # --------------------------------------------------------------- settings
    def _set(self, key: str, value) -> None:
        setattr(self.settings, key, value)
        self.settings.save()

    def _refresh_chip(self) -> None:
        if self.engine.on_gpu:
            gb = self.engine.vram_total_gb()
            self.title.set_chip(f"{self.engine.gpu_name()}   {gb:.0f} GB" if gb >= 9.5
                                else f"{self.engine.gpu_name()}   {gb:.1f} GB")
        elif torch.cuda.is_available():
            self.title.set_chip("Processor only - graphics card is off")
        else:
            self.title.set_chip("No graphics card found - slower")

    def apply_theme(self, key: str) -> None:
        set_theme(key)
        self.settings.theme = key
        self.settings.save()
        app = QApplication.instance()
        if app is not None:
            app.setPalette(_dark_palette())
            app.setStyleSheet(tooltip_qss())
        self.scroll.setStyleSheet(scroll_qss())
        self.queue.restyle()
        restyle_tree(self)
        self.view._invalidate()
        self.update()

    def _refresh_out_label(self) -> None:
        d = self.settings.output_dir
        self.out_label.setFullText(d if d else str(default_download_dir()))

    def _refresh_download_ui(self) -> None:
        job = self.jobs[self.current] if 0 <= self.current < len(self.jobs) else None
        ready = bool(job and job.status == "done" and job.out_path and Path(job.out_path).exists())
        saved = bool(job and job.saved_path and Path(job.saved_path).exists())
        self.btn_download.setVisible(ready and not saved)
        self.btn_show.setVisible(saved)

    def download_job(self, i: int) -> Optional[str]:
        """Copy one finished result from the work folder to the download folder."""
        if not (0 <= i < len(self.jobs)):
            return None
        job = self.jobs[i]
        if job.status != "done" or not job.out_path or not Path(job.out_path).exists():
            return None
        folder = Path(self.settings.output_dir) if self.settings.output_dir else default_download_dir()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            dst = unique_path(folder / Path(job.out_path).name)
            shutil.copy2(job.out_path, dst)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            show_message(self, "Couldn't save the file",
                         f"{e}\n\nYou can pick another folder with Change.")
            return None
        QApplication.restoreOverrideCursor()
        job.saved_path = str(dst)
        return str(dst)

    def on_download(self) -> None:
        dst = self.download_job(self.current)
        if dst:
            self._set_status(f"Saved to {dst}")
            self._refresh_download_ui()

    def download_all(self) -> None:
        n = 0
        last = ""
        for i, j in enumerate(self.jobs):
            if j.status == "done" and not j.saved_path:
                dst = self.download_job(i)
                if dst:
                    n += 1
                    last = dst
        if n:
            self._set_status(f"Saved {n} file{'s' if n != 1 else ''} to {Path(last).parent}")
        self._refresh_download_ui()

    def _show_saved(self) -> None:
        if 0 <= self.current < len(self.jobs) and self.jobs[self.current].saved_path:
            open_in_folder(self.jobs[self.current].saved_path)

    def _on_quality(self, k: str) -> None:
        self._set("quality", k)
        if k != "turbo":
            self._set("photo_quality", k)
        for w in (self.seg_quality, self.seg_quality4):
            if k in w.keys and w.value() != k:
                w.setValue(k)

    def _apply_quality_ui(self, kind: str) -> None:
        """Turbo exists only for videos: show the 4-option row for a video, the normal 3 otherwise."""
        video = kind == "video"
        if not video and self.settings.quality == "turbo":
            pq = self.settings.photo_quality
            self._on_quality(pq if pq in ("maximum", "balanced", "fast") else "maximum")
        self.seg_quality.setVisible(not video)
        self.seg_quality4.setVisible(video)
        for w in (self.seg_quality, self.seg_quality4):
            if self.settings.quality in w.keys and w.value() != self.settings.quality:
                w.setValue(self.settings.quality)

    def _apply_kind_ui(self, kind: str) -> None:
        self.seg_format.setVisible(kind != "video")
        self.seg_vformat.setVisible(kind == "video")
        self._apply_quality_ui(kind)

    def _update_size_info(self) -> None:
        w, h = self._cur_size
        if w <= 0 or h <= 0:
            self.size_info.setText("")
            return
        job = self.jobs[self.current] if 0 <= self.current < len(self.jobs) else None
        ow, oh, _ = plan_output(w, h, self.settings.target, even=bool(job and job.kind == "video"))
        self.size_info.setText(f"{w} \u00d7 {h}   \u2192   {ow} \u00d7 {oh}   ({ow * oh / 1e6:.1f} MP)")

    def choose_output_dir(self) -> None:
        start = self.settings.output_dir or str(default_download_dir())
        d = QFileDialog.getExistingDirectory(self, "Where should downloads go?", start)
        if d:
            self.settings.output_dir = d
            self.settings.save()
            self._refresh_out_label()

    def open_settings_dialog(self) -> None:
        def changed(what: str) -> None:
            if what == "theme":
                self.apply_theme(self.settings.theme)
                return
            if what == "folder":
                self._refresh_out_label()
                return
            if what == "ambient":
                self._bg.setInterval(40 if self.settings.ambient else 160)
                self.update()
                return
            if what == "gpu":
                if self.worker is not None and self.worker.isRunning():
                    show_message(self, "Settings", "This will apply after the current work is finished.")
                    return
                self.engine.set_device(self.settings.use_gpu, self.settings.gpu_index)
                self._refresh_chip()
        open_settings(self, self.settings, self.engine, changed)

    def _warn_no_gpu(self) -> None:
        show_message(
            self, "No graphics card found",
            "YokoEnhancer can still work, but it will be very slow without your NVIDIA graphics card.\n\n"
            "Most likely the CPU version of PyTorch is installed. In a terminal run:\n"
            "pip uninstall torch torchvision -y\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    # ----------------------------------------------------------------- window
    def toggle_max(self) -> None:
        if self.isMaximized():
            self.showNormal()
            self.root.setContentsMargins(self.BORDER, self.BORDER, self.BORDER, self.BORDER)
            self.title.btn_max.setIcon_("max")
        else:
            self.showMaximized()
            self.root.setContentsMargins(0, 0, 0, 0)
            self.title.btn_max.setIcon_("restore")

    def _edges_at(self, pos: QPointF) -> Qt.Edge:
        b = self.BORDER + 3
        x, y, w, h = pos.x(), pos.y(), self.width(), self.height()
        e = Qt.Edge(0)
        if x < b:
            e |= Qt.Edge.LeftEdge
        if x > w - b:
            e |= Qt.Edge.RightEdge
        if y < b:
            e |= Qt.Edge.TopEdge
        if y > h - b:
            e |= Qt.Edge.BottomEdge
        return e

    def mouseMoveEvent(self, ev) -> None:
        if self.isMaximized():
            return
        e = self._edges_at(ev.position())
        L, R, T, Bt = Qt.Edge.LeftEdge, Qt.Edge.RightEdge, Qt.Edge.TopEdge, Qt.Edge.BottomEdge
        if (e & L and e & T) or (e & R and e & Bt):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif (e & R and e & T) or (e & L and e & Bt):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif e & L or e & R:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif e & T or e & Bt:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and not self.isMaximized():
            e = self._edges_at(ev.position())
            if e != Qt.Edge(0):
                wh = self.windowHandle()
                if wh is not None:
                    wh.startSystemResize(e)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect()
        W, H = float(r.width()), float(r.height())
        p.fillRect(r, QColor(*CUR.theme.bg))
        t = time.time()
        blobs = [
            (0.14, 0.18, 640, 0.040, 0.031, 0.0, 26),
            (0.78, 0.30, 560, 0.033, 0.047, 1.7, 20),
            (0.46, 0.92, 720, 0.028, 0.036, 3.1, 24),
            (0.92, 0.88, 460, 0.050, 0.029, 4.4, 16),
        ]
        for bx, by, rad, sx, sy, ph, alpha in blobs:
            cx = (bx + 0.07 * math.sin(t * sx * 2 + ph)) * W
            cy = (by + 0.07 * math.cos(t * sy * 2 + ph)) * H
            g = QRadialGradient(QPointF(cx, cy), rad)
            g.setColorAt(0.0, _rgba_glow(alpha * CUR.theme.blob))
            g.setColorAt(0.55, _rgba_glow(alpha * 0.28 * CUR.theme.blob))
            g.setColorAt(1.0, _rgba_glow(0))
            p.fillRect(r, g)
        if self.settings.ambient:
            try:
                self._ambient.paint(p, W, H)
            except Exception as e:      # never let a drawing slip break the window
                log.warning("Background animation stopped: %s", e)
                self.settings.ambient = False
        vg = QRadialGradient(QPointF(W / 2, H / 2), max(W, H) * 0.75)
        vg.setColorAt(0.5, QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, QColor(0, 0, 0, 170))
        p.fillRect(r, vg)
        if not self.isMaximized():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(warm(34), 1))
            p.drawRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5))
        p.end()

    # -------------------------------------------------------------- drag/drop
    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        paths: List[str] = []
        for u in ev.mimeData().urls():
            if u.isLocalFile():
                paths.append(u.toLocalFile())
        self.add_files(paths)

    # ------------------------------------------------------------------ queue
    def open_dialog(self) -> None:
        start = self.settings.last_dir or str(Path.home() / "Pictures")
        files, _ = QFileDialog.getOpenFileNames(self, "Choose photos or videos", start, self.FILE_FILTER)
        if files:
            self.add_files(files)

    def add_files(self, paths: List[str]) -> None:
        found: List[str] = []
        for pth in paths:
            pp = Path(pth)
            if pp.is_dir():
                for f in sorted(pp.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                        found.append(str(f))
            elif pp.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                found.append(str(pp))
        known = {j.path for j in self.jobs}
        first_new = -1
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for f in found:
                if f in known:
                    continue
                known.add(f)
                kind = "video" if Path(f).suffix.lower() in VIDEO_EXTS else "image"
                job = Job(path=f, kind=kind, name=Path(f).name)
                th = make_thumbnail(f, kind)
                if th is not None:
                    job.thumb = bgr_to_qimage(th)
                self.jobs.append(job)
                it = QListWidgetItem()
                it.setData(Qt.ItemDataRole.UserRole, job)
                it.setSizeHint(QSize(QueueDelegate.W, QueueDelegate.H))
                self.queue.addItem(it)
                if first_new < 0:
                    first_new = len(self.jobs) - 1
                self.settings.last_dir = str(Path(f).parent)
        finally:
            QApplication.restoreOverrideCursor()
        if found:
            self.settings.save()
        if first_new >= 0:
            self.queue.setCurrentRow(first_new)
        elif found:
            self._set_status("Those files are already in the list.")

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.jobs) or self._selecting:
            return
        if row != self.current:
            self._show_job(row)

    def _load_result_image(self, i: int) -> Optional["np.ndarray"]:
        job = self.jobs[i]
        if i in self._result_cache:
            return self._result_cache[i]
        try:
            if job.kind == "video":
                frame, _, _, _ = grab_video_frame(job.out_path, 0.35)
                return frame
            img, _ = imread_any(job.out_path)
            return img
        except Exception as e:
            log.warning("Could not load result: %s", e)
            return None

    def _show_job(self, i: int) -> None:
        job = self.jobs[i]
        self.current = i
        self._preview_rect = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if job.kind == "video":
                frame, fps, (w, h), dur = grab_video_frame(job.path, 0.35)
                before = frame
                job.size, job.duration = (w, h), dur
            else:
                before, _ = imread_any(job.path)
                job.size = (before.shape[1], before.shape[0])
            self.view.set_before(before)
            if job.status == "done" and job.out_path and Path(job.out_path).exists():
                after = self._load_result_image(i)
                if after is not None:
                    self.view.set_after(after)
            self._refresh_download_ui()
            self._cur_size = job.size
            self._apply_kind_ui(job.kind)
            self._update_size_info()
            self._selecting = True
            self.queue.setCurrentRow(i)
            self._selecting = False
            extra = ""
            if job.kind == "video" and job.duration:
                extra = f"   {fmt_time(job.duration)}"
            if not (self.worker and self.worker.isRunning()):
                self._set_status(job.name + extra)
        except Exception as e:
            log.error("Could not open %s: %s", job.path, e)
            self.view.clear()
            show_message(self, "Can't open this file", str(e))
        finally:
            self._selecting = False
            QApplication.restoreOverrideCursor()

    def _queue_menu(self, pos) -> None:
        item = self.queue.itemAt(pos)
        if item is None:
            return
        row = self.queue.row(item)
        job = self.jobs[row]
        m = QMenu(self)
        m.setStyleSheet(menu_qss())
        a_rm = m.addAction("Remove from list")
        a_dl = m.addAction("Download enhanced file") if (job.status == "done" and not job.saved_path) else None
        a_dlall = (m.addAction("Download all finished")
                   if any(j.status == "done" and not j.saved_path for j in self.jobs) else None)
        a_show = m.addAction("Show downloaded file") if job.saved_path else None
        a_orig = m.addAction("Show original")
        act = m.exec(self.queue.viewport().mapToGlobal(pos))
        if act is a_rm:
            self.queue.setCurrentRow(row)
            self.remove_current()
        elif a_dl is not None and act is a_dl:
            dst = self.download_job(row)
            if dst:
                self._set_status(f"Saved to {dst}")
                self._refresh_download_ui()
        elif a_dlall is not None and act is a_dlall:
            self.download_all()
        elif a_show is not None and act is a_show:
            open_in_folder(job.saved_path)
        elif act is a_orig:
            open_in_folder(job.path)

    def _busy(self) -> bool:
        return bool((self.worker and self.worker.isRunning()) or
                    (self.preview_worker and self.preview_worker.isRunning()))

    def remove_current(self) -> None:
        if self._busy() or not (0 <= self.current < len(self.jobs)):
            return
        i = self.current
        self.jobs.pop(i)
        self._result_cache.clear()
        self._selecting = True
        self.queue.takeItem(i)
        self._selecting = False
        if self.jobs:
            self.current = -1
            self.queue.setCurrentRow(min(i, len(self.jobs) - 1))
            self._show_job(min(i, len(self.jobs) - 1))
        else:
            self.current = -1
            self._apply_quality_ui("image")
            self.view.clear()
            self._cur_size = (0, 0)
            self._update_size_info()
            self.btn_show.hide()
            self.btn_download.hide()
            self._set_status("Ready")

    def clear_queue(self) -> None:
        if self._busy():
            return
        self.jobs.clear()
        self._result_cache.clear()
        self._selecting = True
        self.queue.clear()
        self._selecting = False
        self.current = -1
        self._apply_quality_ui("image")
        self.view.clear()
        self._cur_size = (0, 0)
        self._update_size_info()
        self.btn_show.hide()
        self.btn_download.hide()
        self.progress.setValue(0)
        self._set_status("Ready")

    def _refresh_queue(self) -> None:
        self.queue.viewport().update()

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    # ---------------------------------------------------------------- preview
    def on_preview(self) -> None:
        if self._busy():
            return
        if not (0 <= self.current < len(self.jobs)):
            self.open_dialog()
            return
        job = self.jobs[self.current]
        cx, cy = self.view.center_norm()
        W, H = job.size
        if self._preview_rect is not None:
            x0, y0, cw, ch, fw, fh = self._preview_rect
            cx = (x0 + cx * cw) / max(1, fw)
            cy = (y0 + cy * ch) / max(1, fh)
        params = EnhanceParams.from_settings(self.settings)
        self.preview_worker = PreviewWorker(self.engine, job, params, (cx, cy), self)
        self.preview_worker.status.connect(self._on_preview_status)
        self.preview_worker.done.connect(lambda b, a, rect: self._on_preview_done(b, a, rect, W, H))
        self.preview_worker.failed.connect(self._on_preview_failed)
        self.preview_worker.finished.connect(self._on_preview_finished)
        self._set_ui_busy(True, preview=True)
        self.view.set_busy("Previewing")
        self._set_status("Making a quick preview of the middle of the view...")
        self.progress.setValue(-1)
        self.progress.setActive(True)
        self.preview_worker.start()

    def _on_preview_status(self, text: str) -> None:
        self.view.set_busy(text)

    def _on_preview_done(self, before, after, rect, W: int, H: int) -> None:
        self.view.set_pair(before, after)
        self._preview_rect = (rect[0], rect[1], rect[2], rect[3], W, H)
        self._set_status("Preview of a small area. Drag the slider to compare, press Enhance for everything.")

    def _on_preview_failed(self, msg: str) -> None:
        if msg:
            show_message(self, "Preview didn't work", msg)
        self._set_status("Ready")

    def _on_preview_finished(self) -> None:
        self.view.set_busy(None)
        self.progress.setActive(False)
        self.progress.setValue(0)
        self._set_ui_busy(False)

    # ---------------------------------------------------------------- enhance
    def _set_ui_busy(self, busy: bool, preview: bool = False) -> None:
        self.btn_preview.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        if preview:
            self.btn_go.setEnabled(not busy)
        else:
            self.btn_go.setEnabled(True)
            if busy:
                self.btn_go.setLabel("Stop", "stop")
            else:
                self.btn_go.setLabel("Enhance", "spark")

    def on_enhance(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self._set_status("Stopping...")
            return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            return
        if not self.jobs:
            self.open_dialog()
            return
        todo = [(i, j) for i, j in enumerate(self.jobs) if j.status != "done"]
        if not todo and 0 <= self.current < len(self.jobs):
            todo = [(self.current, self.jobs[self.current])]
        if not todo:
            return
        if self.settings.quality not in ("fast", "turbo") and any(j.kind == "video" for _, j in todo):
            ans = show_message(self, "Video speed",
                               "Maximum is built for photos and is very slow for video.\n\n"
                               "Turbo is made for video: many times quicker, with a very similar natural look.",
                               ("Use Turbo", "Keep my choice", "Cancel"), 0)
            if ans == 0:
                self._on_quality("turbo")
            elif ans != 1:
                return
        for _, j in todo:
            j.status = "queued"
            j.progress = 0.0
            j.message = ""
        self._errors = []
        self._done_count = 0
        self._refresh_queue()
        params = EnhanceParams.from_settings(self.settings)
        w = EnhanceWorker(self.engine, todo, params, self.settings, self)
        self.worker = w
        w.status.connect(self._on_worker_status)
        w.overall.connect(self.progress.setValue)
        w.jobStarted.connect(self._on_job_started)
        w.jobProgress.connect(self._on_job_progress)
        w.jobDone.connect(self._on_job_done)
        w.jobFailed.connect(self._on_job_failed)
        w.fatal.connect(self._on_fatal)
        w.allDone.connect(self._on_all_done)
        self.btn_show.hide()
        self.btn_download.hide()
        self._set_ui_busy(True)
        self.progress.setValue(0)
        self.progress.setActive(True)
        self._set_status("Getting ready...")
        self.view.set_busy("Getting ready")
        w.start()

    def _on_worker_status(self, text: str) -> None:
        self._set_status(text)
        short = text.split("  -  ")[0]
        self.view.set_busy(short)

    def _on_job_started(self, i: int) -> None:
        if 0 <= i < len(self.jobs):
            self.jobs[i].status = "working"
            self.jobs[i].progress = 0.0
            self._refresh_queue()
            if i != self.current:
                self._show_job(i)
                self.view.set_busy("Enhancing")

    def _on_job_progress(self, i: int, f: float) -> None:
        if 0 <= i < len(self.jobs):
            self.jobs[i].progress = f
            self._refresh_queue()

    def _on_job_done(self, i: int, path: str, arr) -> None:
        if not (0 <= i < len(self.jobs)):
            return
        job = self.jobs[i]
        job.status = "done"
        job.progress = 1.0
        job.out_path = path
        self._done_count += 1
        self._last_out = path
        if arr is not None:
            self._result_cache[i] = arr
            while len(self._result_cache) > 4:
                self._result_cache.pop(next(iter(self._result_cache)))
        self._refresh_queue()
        if i == self.current:
            self._show_job(i)
            self.view.set_busy("Done")

    def _on_job_failed(self, i: int, msg: str) -> None:
        if not (0 <= i < len(self.jobs)):
            return
        job = self.jobs[i]
        if msg == "__cancelled__":
            job.status = "queued"
            job.progress = 0.0
        else:
            job.status = "error"
            job.message = msg
            self._errors.append(f"{job.name}: {msg}")
        self._refresh_queue()

    def _on_fatal(self, msg: str) -> None:
        self._errors.append(msg)

    def _on_all_done(self, cancelled: bool) -> None:
        self.progress.setActive(False)
        self.view.set_busy(None)
        self._set_ui_busy(False)
        if cancelled:
            self.progress.setValue(0)
            self._set_status("Stopped.")
        elif self._errors:
            self.progress.setValue(0)
            self._set_status("Finished with a problem.")
        else:
            self.progress.setValue(1.0)
            n = self._done_count
            self._set_status(f"Done - {n} file{'s' if n != 1 else ''} ready. Press Download to save.")
        self._refresh_download_ui()
        for j in self.jobs:
            if j.status == "working":
                j.status = "queued"
        self._refresh_queue()
        if self._errors:
            show_message(self, "Couldn't finish", "\n\n".join(self._errors[:3]))
            self._errors = []
        if 0 <= self.current < len(self.jobs) and self.jobs[self.current].status == "done" \
                and not self.view.has_result():
            self._show_job(self.current)

    # ------------------------------------------------------------------ close
    def closeEvent(self, ev) -> None:
        for w in (self.worker, self.preview_worker):
            if w is not None and w.isRunning():
                try:
                    w.stop() if isinstance(w, EnhanceWorker) else w.cancel_event.set()
                except Exception:
                    pass
                w.wait(4000)
        self.settings.save()
        super().closeEvent(ev)


# ============================================================================
#  ENTRY POINT
# ============================================================================
def _dark_palette() -> QtGui.QPalette:
    pal = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    t = CUR.theme
    txt = QColor(*t.text)

    def dk(v: int) -> QColor:
        return QColor(min(255, t.bg[0] + v), min(255, t.bg[1] + v), min(255, t.bg[2] + v))

    pal.setColor(R.Window, dk(6))
    pal.setColor(R.WindowText, txt)
    pal.setColor(R.Base, dk(8))
    pal.setColor(R.AlternateBase, dk(14))
    pal.setColor(R.Text, txt)
    pal.setColor(R.Button, dk(14))
    pal.setColor(R.ButtonText, txt)
    pal.setColor(R.Highlight, txt)
    pal.setColor(R.HighlightedText, dk(6))
    pal.setColor(R.ToolTipBase, dk(12))
    pal.setColor(R.ToolTipText, txt)
    return pal


def main() -> int:
    set_theme(Settings.load().theme)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("YokoEnhancer.App.1")
        except Exception:
            pass
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setFont(ui_font(13))
    app.setWindowIcon(make_app_icon())
    app.setStyleSheet(tooltip_qss())
    win = YokoWindow()
    win.show()
    args = [a for a in sys.argv[1:] if Path(a).exists()]
    if args:
        QTimer.singleShot(200, lambda: win.add_files(args))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
