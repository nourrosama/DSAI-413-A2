"""
medgemma.py — MedGemma loader and inference wrapper.

Model: google/medgemma-4b-it  (instruction-tuned, 4B params)
Accepts image + text prompt → generates text.
Used for both Mode 1 (report generation) and Mode 2 (QA answering).

Prerequisites:
  - Accept the MedGemma license on HuggingFace:
    https://huggingface.co/google/medgemma-4b-it
  - Set HF_TOKEN environment variable or run `huggingface-cli login`
"""

import os
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from src.config import (
    MEDGEMMA_MODEL_ID,
    MEDGEMMA_DTYPE,
    MEDGEMMA_MAX_TOKENS,
    MEDGEMMA_QA_MAX_TOKENS,
    REPORT_GEN_PROMPT,
    QA_ANSWER_PROMPT_TEMPLATE,
)


class MedGemmaModel:
    """
    Wrapper around MedGemma for image-conditioned text generation.

    Usage
    -----
    >>> model = MedGemmaModel()
    >>> model.load()
    >>> report = model.generate_report(pil_image)
    >>> answer = model.answer_question(pil_image, "Is there cardiomegaly?", context="...")
    """

    def __init__(
        self,
        model_id: str = MEDGEMMA_MODEL_ID,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the model and processor. Call once before inference."""
        print(f"Loading MedGemma from '{self.model_id}' …")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        quantization_config = None
        if self.load_in_4bit and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            torch_dtype=dtype if quantization_config is None else None,
            device_map="auto" if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        print("MedGemma loaded ✓")

    def _check_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

    # ── Internal generation ───────────────────────────────────────────────────

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Core generation — takes a PIL image and a text prompt."""
        self._check_loaded()

        # Build the conversation in chat format expected by instruction-tuned models
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic for reproducibility
                temperature=1.0,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_report(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = MEDGEMMA_MAX_TOKENS,
    ) -> str:
        """
        Mode 1 — Generate a structured radiology report from a chest X-ray.

        Parameters
        ----------
        image          : PIL Image (RGB)
        prompt         : Override the default report generation prompt if needed.
        max_new_tokens : Token budget for the generated report.

        Returns
        -------
        str  Structured report text (Findings / Impression).
        """
        prompt = prompt or REPORT_GEN_PROMPT
        raw = self._generate(image, prompt, max_new_tokens)
        return _enforce_report_structure(raw)

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        context: str = "",
        max_new_tokens: int = MEDGEMMA_QA_MAX_TOKENS,
    ) -> str:
        """
        Mode 2 — Answer a clinical question about a chest X-ray.

        Parameters
        ----------
        image     : PIL Image (RGB)
        question  : Natural-language clinical question.
        context   : Retrieved report text to ground the answer (RAG context).

        Returns
        -------
        str  Clinical answer.
        """
        prompt = QA_ANSWER_PROMPT_TEMPLATE.format(
            context=context if context else "No additional context available.",
            question=question,
        )
        return self._generate(image, prompt, max_new_tokens)

    def batch_generate_reports(
        self,
        images: List[Image.Image],
        verbose: bool = True,
    ) -> List[str]:
        """Generate reports for a list of images sequentially."""
        reports = []
        for i, img in enumerate(images):
            if verbose:
                print(f"  [{i+1}/{len(images)}] Generating report …", end="\r")
            reports.append(self.generate_report(img))
        if verbose:
            print(f"  Generated {len(reports)} reports ✓")
        return reports


# ─── Post-processing ──────────────────────────────────────────────────────────

def _enforce_report_structure(text: str) -> str:
    """
    Ensure the generated text contains FINDINGS and IMPRESSION sections.
    If the model already produced them, return as-is.
    Otherwise wrap the output in a minimal structure.
    """
    upper = text.upper()
    has_findings   = "FINDINGS" in upper
    has_impression = "IMPRESSION" in upper

    if has_findings and has_impression:
        return text

    # Fallback: treat the whole output as findings
    structured = "FINDINGS:\n"
    structured += text.strip()
    if not has_impression:
        structured += "\n\nIMPRESSION:\nPlease refer to the findings above."
    return structured
