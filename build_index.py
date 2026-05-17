"""
build_index.py - Build and save a ColPali FAISS index for the Streamlit demo.

ColPali is the mandatory retrieval model for this assignment. Even on CPU it
works on a small set of images (~5-10 seconds per image to embed). For 8
images that's ~1 minute total.

Saves to the default FAISS_INDEX_PATH / FAISS_META_PATH so the Streamlit
app's `load_index()` finds it automatically.

Run:
    python build_index.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_image
from src.models.colpali import ColPaliModel
from src.retrieval import build_index_from_images
from src import config as cfg


def main():
    sample_dir = ROOT / "data" / "sample_images"
    if not sample_dir.exists():
        sys.exit(f"No folder at {sample_dir}. Put some X-ray PNG/JPG files there first.")

    files = sorted([str(p) for p in sample_dir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not files:
        sys.exit(f"No images in {sample_dir}. Drop 8+ X-ray PNGs/JPGs there.")

    # Limit to 8 on CPU so it finishes in a reasonable time
    files = files[:8]
    print(f"Indexing {len(files)} images from {sample_dir}")

    print("Loading images ...")
    images = [load_image(p) for p in files]
    reports = [f"Chest X-ray sample {Path(p).stem}: standard frontal projection."
               for p in files]

    print("\nLoading ColPali (first run downloads ~3 GB; subsequent runs are fast) ...")
    t0 = time.time()
    colpali = ColPaliModel()
    colpali.load()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("\nEmbedding images ...")
    t0 = time.time()
    idx = build_index_from_images(
        images=images,
        reports=reports,
        image_paths=files,
        backend="colpali",
        save=False,        # save manually below
        model=colpali,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # Save at the default paths so Streamlit's `load_index()` finds it
    os.makedirs(os.path.dirname(cfg.FAISS_INDEX_PATH), exist_ok=True)
    idx.save(cfg.FAISS_INDEX_PATH, cfg.FAISS_META_PATH)
    print(f"\nSaved index    -> {cfg.FAISS_INDEX_PATH}")
    print(f"Saved metadata -> {cfg.FAISS_META_PATH}")
    print(f"\nNow restart Streamlit and make sure the sidebar")
    print(f"'Retrieval backend' is set to 'colpali'.")


if __name__ == "__main__":
    main()
