"""
preprocessing.py — Image loading, conversion, and normalization utilities.

Supports:
  - JPEG / PNG  (via Pillow)
  - DICOM (.dcm) (via pydicom)
  - Batch loading from a list of paths or a directory
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

# Optional DICOM support — graceful fallback if pydicom not installed
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
# MedGemma and most HuggingFace VLMs expect 224×224 or 336×336 RGB images.
# The AutoProcessor handles the final normalization; we just standardize size here.
DEFAULT_SIZE: Tuple[int, int] = (224, 224)
DICOM_EXTENSIONS = {".dcm", ".dicom"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


# ─── Core loaders ─────────────────────────────────────────────────────────────

def load_image(path: Union[str, Path], size: Tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    """
    Load any supported image file and return a resized RGB PIL Image.

    Parameters
    ----------
    path : str | Path
        Path to the image file (PNG, JPG, or DICOM).
    size : (width, height)
        Target size. Defaults to DEFAULT_SIZE.

    Returns
    -------
    PIL.Image.Image  in RGB mode, resized to `size`.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in DICOM_EXTENSIONS:
        img = _load_dicom(path)
    elif ext in IMAGE_EXTENSIONS:
        img = Image.open(path).convert("RGB")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    img = img.resize(size, Image.LANCZOS)
    return img


def _load_dicom(path: Path) -> Image.Image:
    """Convert a DICOM file to an 8-bit grayscale PIL Image, then to RGB."""
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM support. Install it with: pip install pydicom")

    ds = pydicom.dcmread(str(path))
    pixel_array = ds.pixel_array.astype(np.float32)

    # Normalize to 0–255
    pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
    if pixel_max > pixel_min:
        pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    img = Image.fromarray(pixel_array.astype(np.uint8), mode="L")
    return img.convert("RGB")


# ─── Batch loading ────────────────────────────────────────────────────────────

def load_images_from_paths(
    paths: List[Union[str, Path]],
    size: Tuple[int, int] = DEFAULT_SIZE,
    verbose: bool = False,
) -> List[Image.Image]:
    """
    Load a list of image paths into PIL Images.

    Skips files that fail to load and prints a warning.
    """
    images = []
    for p in paths:
        try:
            images.append(load_image(p, size=size))
            if verbose:
                print(f"  ✓ {p}")
        except Exception as e:
            print(f"  ✗ Could not load {p}: {e}")
    return images


def load_images_from_directory(
    directory: Union[str, Path],
    size: Tuple[int, int] = DEFAULT_SIZE,
    max_images: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[Image.Image], List[Path]]:
    """
    Recursively load all supported images from a directory.

    Returns
    -------
    (images, paths) — parallel lists of PIL Images and their source paths.
    """
    directory = Path(directory)
    all_exts = IMAGE_EXTENSIONS | DICOM_EXTENSIONS
    all_paths = sorted([p for p in directory.rglob("*") if p.suffix.lower() in all_exts])

    if max_images is not None:
        all_paths = all_paths[:max_images]

    images = load_images_from_paths(all_paths, size=size, verbose=verbose)
    return images, all_paths[:len(images)]


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def load_mimic_subset(
    csv_path: str,
    img_col: str,
    text_col: str,
    subset_size: Optional[int] = None,
    image_base_dir: Optional[str] = None,
    size: Tuple[int, int] = DEFAULT_SIZE,
    seed: int = 42,
) -> Tuple[List[Image.Image], List[str], List[str]]:
    """
    Load a subset of MIMIC-CXR images and their ground-truth reports.

    Handles the dataset format where both `image` and `text` columns store
    stringified Python lists, e.g.:
      image -> "['files/p10/.../img1.jpg', 'files/p10/.../img2.jpg']"
      text  -> "['Findings: No focal consolidation...']"

    One image-report pair is produced per image path found. If a row has
    multiple images, each image is paired with the same report text.

    Parameters
    ----------
    csv_path       : path to the metadata CSV
    img_col        : column containing stringified list of relative image paths
    text_col       : column containing stringified list with the report text
    subset_size    : how many pairs to sample (None = all)
    image_base_dir : prepended to each image path from the CSV
    size           : resize target for loaded images

    Returns
    -------
    (images, reports, img_paths)
    """
    import ast
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[img_col, text_col])

    # ── Step 1: scan disk once to build a set of all available image paths ────
    # This avoids trying to open thousands of missing files one by one.
    print("Scanning available images on disk …")
    available = set()
    if image_base_dir and os.path.exists(image_base_dir):
        for root, _, files in os.walk(image_base_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full = os.path.join(root, fname)
                    # Store as relative path from image_base_dir (forward slashes)
                    rel = os.path.relpath(full, image_base_dir).replace("\\", "/")
                    available.add(rel)
    print(f"Found {len(available):,} images on disk.")

    images, reports, valid_paths = [], [], []

    for _, row in df.iterrows():
        # ── Parse image paths (stringified list) ──────────────────────────────
        try:
            img_paths = ast.literal_eval(str(row[img_col]))
            if not isinstance(img_paths, list):
                img_paths = [img_paths]
        except Exception:
            img_paths = [str(row[img_col]).strip()]

        # ── Parse report text (stringified list — take first element) ─────────
        try:
            text_list = ast.literal_eval(str(row[text_col]))
            if isinstance(text_list, list):
                report_text = str(text_list[0]).strip()
            else:
                report_text = str(text_list).strip()
        except Exception:
            report_text = str(row[text_col]).strip()

        if not report_text:
            continue

        # ── Only load images that exist on disk ───────────────────────────────
        for rel_path in img_paths:
            rel_norm = rel_path.replace("\\", "/")
            if rel_norm not in available:
                continue  # skip silently — not downloaded

            full_path = os.path.join(image_base_dir, rel_path) if image_base_dir else rel_path
            try:
                img = load_image(full_path, size=size)
                images.append(img)
                reports.append(report_text)
                valid_paths.append(full_path)
            except Exception as e:
                print(f"  ✗ Could not open {full_path}: {e}")

        # Stop early if we have enough
        if subset_size is not None and len(images) >= subset_size:
            break

    # Trim to exact subset size
    if subset_size is not None:
        images      = images[:subset_size]
        reports     = reports[:subset_size]
        valid_paths = valid_paths[:subset_size]

    print(f"Loaded {len(images)} image-report pairs from MIMIC-CXR.")
    return images, reports, valid_paths


# ─── Normalization helpers (used outside HuggingFace AutoProcessor) ───────────

def normalize_image_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to a float32 numpy array normalized to [0, 1].
    Shape: (H, W, 3)
    """
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return arr


def pil_to_tensor(img: Image.Image):
    """
    Convert PIL Image to a PyTorch float tensor of shape (3, H, W) in [0, 1].
    Requires torch.
    """
    import torch
    arr = normalize_image_array(img)
    return torch.from_numpy(arr).permute(2, 0, 1)
