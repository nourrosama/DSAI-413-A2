"""
mode1_report_gen.py — Mode 1: Report Generation Pipeline

Input  : A chest X-ray image (PIL Image or file path)
Output : A structured medical report with FINDINGS and IMPRESSION sections

Pipeline
--------
1. Load and preprocess the image.
2. Generate a report using MedGemma.
3. (Optional) Compute CLIP image-text alignment score as a comparison baseline.
4. Evaluate against ground truth if provided (BLEU / ROUGE-L / BERTScore).

Both MedGemma (primary generator) and CLIP (embedding baseline) are exercised
here so the Mode 1 comparison section is well supported.
"""

from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.config import MEDGEMMA_MAX_TOKENS, REPORT_GEN_PROMPT
from src.preprocessing import load_image


# ─── Main pipeline class ──────────────────────────────────────────────────────

class ReportGenerationPipeline:
    """
    End-to-end Mode 1 pipeline.

    Usage
    -----
    >>> pipeline = ReportGenerationPipeline()
    >>> pipeline.load_models()
    >>> result = pipeline.run(image_path="path/to/xray.png")
    >>> print(result["report"])
    """

    def __init__(
        self,
        use_medgemma: bool = True,
        use_clip: bool = True,
        medgemma_load_in_4bit: bool = True,
    ):
        self.use_medgemma = use_medgemma
        self.use_clip = use_clip
        self.medgemma_load_in_4bit = medgemma_load_in_4bit

        self._medgemma = None
        self._clip = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load all requested models. Call once before inference."""
        if self.use_medgemma:
            from src.models.medgemma import MedGemmaModel
            self._medgemma = MedGemmaModel(load_in_4bit=self.medgemma_load_in_4bit)
            self._medgemma.load()

        if self.use_clip:
            from src.models.clip_model import CLIPEncoder
            self._clip = CLIPEncoder()
            self._clip.load()

    # ── Single image inference ─────────────────────────────────────────────────

    def run(
        self,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        ground_truth_report: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict:
        """
        Run the full Mode 1 pipeline on a single image.

        Parameters
        ----------
        image                 : PIL Image (if already loaded).
        image_path            : Path to image file (used if image is None).
        ground_truth_report   : Reference report for metric computation (optional).
        prompt                : Override the default MedGemma prompt (optional).

        Returns
        -------
        dict with keys:
          "report"           : Generated structured report (str)
          "clip_alignment"   : CLIP image-text cosine similarity score (float | None)
          "metrics"          : BLEU/ROUGE/BERTScore dict (if ground_truth provided)
          "image"            : The PIL Image used
        """
        # 1. Load image
        if image is None and image_path is None:
            raise ValueError("Provide either `image` or `image_path`.")
        if image is None:
            image = load_image(image_path)

        result = {"image": image, "report": None, "clip_alignment": None, "metrics": None}

        # 2. MedGemma report generation
        if self.use_medgemma and self._medgemma is not None:
            result["report"] = self._medgemma.generate_report(
                image, prompt=prompt, max_new_tokens=MEDGEMMA_MAX_TOKENS
            )

        # 3. CLIP alignment score (how well the generated report matches the image)
        if self.use_clip and self._clip is not None and result["report"]:
            img_emb = self._clip.embed_single_image(image)       # (1, D)
            txt_emb = self._clip.embed_texts([result["report"]]) # (1, D)
            sim = float((img_emb * txt_emb).sum())               # cosine (already normalized)
            result["clip_alignment"] = round(sim, 4)

        # 4. Evaluation metrics
        if ground_truth_report and result["report"]:
            from src.evaluation import compute_report_metrics
            result["metrics"] = compute_report_metrics(
                [result["report"]], [ground_truth_report]
            )

        return result

    # ── Batch inference ───────────────────────────────────────────────────────

    def run_batch(
        self,
        images: List[Image.Image],
        ground_truth_reports: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run the pipeline on a list of images.

        Returns
        -------
        List of result dicts (same structure as run()).
        """
        results = []
        for i, img in enumerate(images):
            if verbose:
                print(f"  [{i+1}/{len(images)}] Processing …", end="\r")
            gt = ground_truth_reports[i] if ground_truth_reports else None
            results.append(self.run(image=img, ground_truth_report=gt))

        if verbose:
            print(f"\n  Processed {len(results)} images.")

        return results

    # ── Comparison helper ─────────────────────────────────────────────────────

    def compare_prompts(
        self,
        image: Image.Image,
        prompts: Dict[str, str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Run MedGemma with multiple different prompts and compare output quality.
        Useful for the Mode 1 prompt ablation experiment.

        Parameters
        ----------
        prompts : {"label": "prompt text", ...}

        Returns
        -------
        {"label": result_dict, ...}
        """
        if self._medgemma is None:
            raise RuntimeError("MedGemma not loaded.")

        comparison = {}
        for label, prompt in prompts.items():
            print(f"  Running prompt variant: '{label}' …")
            report = self._medgemma.generate_report(image, prompt=prompt)
            entry = {"report": report}
            if ground_truth:
                from src.evaluation import compute_report_metrics
                entry["metrics"] = compute_report_metrics([report], [ground_truth])
            comparison[label] = entry

        return comparison


# ─── Convenience function ─────────────────────────────────────────────────────

def generate_report_quick(
    image_path: str,
    load_in_4bit: bool = True,
) -> str:
    """
    One-liner convenience function — loads model, runs inference, returns report.
    For notebook / quick testing use. Not memory-efficient for repeated calls.
    """
    pipeline = ReportGenerationPipeline(use_clip=False, medgemma_load_in_4bit=load_in_4bit)
    pipeline.load_models()
    result = pipeline.run(image_path=image_path)
    return result["report"]


# ─── Prompt variants for comparison experiment ────────────────────────────────

PROMPT_VARIANTS = {
    "default": REPORT_GEN_PROMPT,

    "brief": (
        "You are a radiologist. Look at this chest X-ray and write a brief report "
        "with FINDINGS and IMPRESSION sections."
    ),

    "detailed": (
        "You are a senior radiologist reviewing a chest X-ray. Provide a comprehensive "
        "radiology report. Under FINDINGS, describe: lung fields (opacity, consolidation, "
        "infiltrates, nodules), cardiac silhouette (size, contour), mediastinum (width, "
        "tracheal position), pleural spaces (effusion, pneumothorax), bones (fractures, "
        "osteopenia), and any tubes or lines. Under IMPRESSION, give a concise summary "
        "with differential diagnosis if appropriate."
    ),

    "few_shot": (
        "You are a radiologist. Here is an example of a good report:\n\n"
        "FINDINGS:\nThe lungs are clear bilaterally. No focal consolidation, pleural "
        "effusion, or pneumothorax identified. The cardiac silhouette is normal in size. "
        "The mediastinum is unremarkable.\n\n"
        "IMPRESSION:\nNormal chest X-ray.\n\n"
        "---\n\n"
        "Now analyze the provided chest X-ray and generate a similarly structured report."
    ),
}
