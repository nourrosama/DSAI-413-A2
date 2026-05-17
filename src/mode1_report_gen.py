"""
mode1_report_gen.py - Mode 1: Report Generation Pipeline

Input  : A chest X-ray image (PIL Image or file path)
Output : A structured medical report with FINDINGS and IMPRESSION sections

Backend selection
-----------------
The generator is picked at .load_models() time based on env var USE_GROQ_VISION:
  USE_GROQ_VISION=1  -> Groq cloud (Llama 4 Scout vision) - no GPU required
  unset / 0          -> MedGemma 4B local (requires CUDA GPU + bf16)

Pipeline
--------
1. Load and preprocess the image.
2. Generate a report using the selected backend.
3. (Optional) Compute CLIP image-text alignment as a comparison baseline.
4. (Optional) Evaluate against ground truth (BLEU / ROUGE-L / BERTScore).
"""

import os
from typing import Dict, List, Optional

from PIL import Image

from src.config import MEDGEMMA_MAX_TOKENS, REPORT_GEN_PROMPT
from src.preprocessing import load_image


def _use_groq() -> bool:
    """True if env var USE_GROQ_VISION is set to "1" / "true" / "yes"."""
    return os.environ.get("USE_GROQ_VISION", "0").lower() in ("1", "true", "yes")


class ReportGenerationPipeline:
    """End-to-end Mode 1 pipeline (Report Generation)."""

    def __init__(
        self,
        use_medgemma: bool = True,
        use_clip: bool = True,
        medgemma_load_in_4bit: bool = False,
    ):
        self.use_medgemma = use_medgemma
        self.use_clip = use_clip
        self.medgemma_load_in_4bit = medgemma_load_in_4bit

        self._medgemma = None      # MedGemma or Groq instance (name kept for API parity)
        self._clip = None
        self._backend = "unset"

    # -- Model loading ---------------------------------------------------------

    def load_models(self) -> None:
        """Load all requested models. Call once before inference."""
        if self.use_medgemma:
            if _use_groq():
                from src.models.groq_vision import GroqVisionModel
                self._medgemma = GroqVisionModel()
                self._backend = "groq"
            else:
                from src.models.medgemma import MedGemmaModel
                self._medgemma = MedGemmaModel(load_in_4bit=self.medgemma_load_in_4bit)
                self._backend = "medgemma"
            self._medgemma.load()
            print(f"Mode 1 generator backend: {self._backend}")

        if self.use_clip:
            from src.models.clip_model import CLIPEncoder
            self._clip = CLIPEncoder()
            self._clip.load()

    # -- Single image inference -----------------------------------------------

    def run(
        self,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        ground_truth_report: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict:
        """
        Run the full Mode 1 pipeline on a single image.

        Returns a dict with keys: report, clip_alignment, metrics, image, backend.
        """
        if image is None and image_path is None:
            raise ValueError("Provide either `image` or `image_path`.")
        if image is None:
            image = load_image(image_path)

        result = {
            "image": image,
            "report": None,
            "clip_alignment": None,
            "metrics": None,
            "backend": self._backend,
        }

        if self.use_medgemma and self._medgemma is not None:
            result["report"] = self._medgemma.generate_report(
                image, prompt=prompt, max_new_tokens=MEDGEMMA_MAX_TOKENS
            )

        if self.use_clip and self._clip is not None and result["report"]:
            img_emb = self._clip.embed_single_image(image)
            txt_emb = self._clip.embed_texts([result["report"]])
            sim = float((img_emb * txt_emb).sum())
            result["clip_alignment"] = round(sim, 4)

        if ground_truth_report and result["report"]:
            from src.evaluation import compute_report_metrics
            result["metrics"] = compute_report_metrics(
                [result["report"]], [ground_truth_report]
            )

        return result

    # -- Batch inference -------------------------------------------------------

    def run_batch(
        self,
        images: List[Image.Image],
        ground_truth_reports: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """Run the pipeline on a list of images sequentially."""
        results = []
        for i, img in enumerate(images):
            if verbose:
                print(f"  [{i+1}/{len(images)}] Processing ...", end="\r")
            gt = ground_truth_reports[i] if ground_truth_reports else None
            results.append(self.run(image=img, ground_truth_report=gt))
        if verbose:
            print(f"\n  Processed {len(results)} images.")
        return results

    # -- Prompt-variant comparison helper --------------------------------------

    def compare_prompts(
        self,
        image: Image.Image,
        prompts: Dict[str, str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Run the generator with several prompts and compare outputs."""
        if self._medgemma is None:
            raise RuntimeError("Generator not loaded. Call load_models() first.")

        comparison = {}
        for label, prompt in prompts.items():
            print(f"  Running prompt variant: '{label}' ...")
            report = self._medgemma.generate_report(image, prompt=prompt)
            entry = {"report": report}
            if ground_truth:
                from src.evaluation import compute_report_metrics
                entry["metrics"] = compute_report_metrics([report], [ground_truth])
            comparison[label] = entry
        return comparison


# -- Convenience function -----------------------------------------------------

def generate_report_quick(image_path: str, load_in_4bit: bool = False) -> str:
    """One-liner convenience function for notebooks."""
    pipeline = ReportGenerationPipeline(use_clip=False, medgemma_load_in_4bit=load_in_4bit)
    pipeline.load_models()
    return pipeline.run(image_path=image_path)["report"]


# -- Prompt variants for comparison experiment --------------------------------

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
