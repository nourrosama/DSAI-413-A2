"""
run_local.py - Single-file end-to-end runner for both modes, CPU-only, minimal data.

Usage:
    # 1. Make sure GROQ_API_KEY is set
    $env:GROQ_API_KEY = "gsk_..."
    $env:USE_GROQ_VISION = "1"

    # 2. Make sure you have a few X-ray images in data/sample_images/
    #    (any chest X-ray PNG/JPG files - 8-10 is enough)

    # 3. Run
    python run_local.py

This produces:
    - Mode 1 generated reports for each sample image
    - A small ColPali FAISS index over the same images
    - Mode 2 RAG answers for a few example questions
    - report/local_results.json with everything
"""

import json
import os
import sys
import time
from pathlib import Path

# Make sure src/ is importable when run from repo root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Force Groq backend for the local CPU-only run
os.environ.setdefault("USE_GROQ_VISION", "1")

# How many images / queries to process. Keep small for CPU.
N_IMAGES = 8
N_QA_PROBE = 3


def main():
    print("=" * 70)
    print("Local end-to-end runner (CPU + Groq Vision)")
    print("=" * 70)

    # --- 0. Sanity checks ---------------------------------------------------
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("ERROR: GROQ_API_KEY env var not set.\n"
                 "Run: $env:GROQ_API_KEY = 'gsk_your_key'")

    sample_dir = ROOT / "data" / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted([p for p in sample_dir.glob("*")
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

    if len(img_files) < N_IMAGES:
        print(f"WARNING: only {len(img_files)} images in data/sample_images/.")
        print("        Place at least 8 chest X-ray PNG/JPG files there and rerun.")
        if len(img_files) == 0:
            sys.exit("No images found. Aborting.")

    img_files = img_files[:N_IMAGES]
    print(f"\nUsing {len(img_files)} images from {sample_dir}")

    # --- 1. Load images -----------------------------------------------------
    from src.preprocessing import load_image
    images = [load_image(p) for p in img_files]
    image_paths = [str(p) for p in img_files]
    # Use the filename as a stand-in "report" (you can edit later)
    fake_reports = [f"Reference report for {p.name}: chest X-ray of typical anatomy."
                    for p in img_files]

    # --- 2. MODE 1 - Report generation -------------------------------------
    print("\n" + "-" * 70)
    print("MODE 1 - Report Generation")
    print("-" * 70)
    from src.mode1_report_gen import ReportGenerationPipeline

    pipe1 = ReportGenerationPipeline(use_clip=False)   # skip CLIP to save time
    pipe1.load_models()

    mode1_results = []
    t0 = time.time()
    for i, img in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_files[i].name}")
        r = pipe1.run(image=img)
        print(r["report"][:400])
        mode1_results.append({
            "image": img_files[i].name,
            "report": r["report"],
            "backend": r["backend"],
        })
    print(f"\nMode 1 done in {time.time()-t0:.1f}s")

    # --- 3. Build a small retrieval index ----------------------------------
    print("\n" + "-" * 70)
    print("Building CLIP retrieval index (fast on CPU)")
    print("-" * 70)
    from src.models.clip_model import CLIPEncoder
    from src.retrieval import build_index_from_images

    clip_enc = CLIPEncoder()
    clip_enc.load()

    index = build_index_from_images(
        images=images,
        reports=fake_reports,
        image_paths=image_paths,
        backend="clip",
        save=False,
        model=clip_enc,
    )

    # --- 4. MODE 2 - QA (RAG) ---------------------------------------------
    print("\n" + "-" * 70)
    print("MODE 2 - QA (RAG)")
    print("-" * 70)
    from src.mode2_qa import QAPipeline

    qa = QAPipeline(index=index, retrieval_backend="clip", top_k=3)
    # Don't re-load the generator (reuse Mode 1's)
    qa._medgemma = pipe1._medgemma
    qa._backend = pipe1._backend
    qa._clip = clip_enc
    print("QA pipeline ready (models reused)")

    sample_questions = [
        "Is there any evidence of pleural effusion?",
        "Are the lung fields clear?",
        "Is the cardiac silhouette enlarged?",
    ]

    mode2_results = []
    t0 = time.time()
    for q in sample_questions[:N_QA_PROBE]:
        # Use the first image as the query
        r = qa.run(image=images[0], question=q)
        print(f"\nQ: {q}")
        print(f"A: {r['answer'][:300]}")
        mode2_results.append({
            "question": q,
            "answer": r["answer"],
            "retrieval_backend": r["retrieval_backend"],
            "generator_backend": r["generator_backend"],
            "top_retrieved": [
                {"image": Path(x["image_path"]).name, "score": round(x["score"], 4)}
                for x in r["retrieved_results"]
            ],
        })
    print(f"\nMode 2 done in {time.time()-t0:.1f}s")

    # --- 5. Save -----------------------------------------------------------
    out_dir = ROOT / "report"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "local_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode1": mode1_results,
            "mode2": mode2_results,
            "n_images": len(images),
            "generator_backend": pipe1._backend,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\nAll done. To launch the demo: streamlit run app/app.py")


if __name__ == "__main__":
    main()
