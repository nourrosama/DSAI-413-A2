"""
groq_vision.py - Drop-in replacement for MedGemma using Groq's vision API.

For local deployment without a CUDA GPU, MedGemma 4B is impractical
(15+ minutes per generation on CPU). This wrapper calls a vision-capable
Llama model on Groq's cloud, exposing the same .load(), .generate_report(),
.answer_question() interface as MedGemmaModel.

Used by Mode 1 and Mode 2 pipelines when env var USE_GROQ_VISION=1.

Prerequisites:
  - Free Groq account: https://console.groq.com
  - Set GROQ_API_KEY environment variable
"""

import base64
import io
import os
from typing import List, Optional

from PIL import Image

from src.config import REPORT_GEN_PROMPT, QA_ANSWER_PROMPT_TEMPLATE


# Vision-capable models available on Groq:
#   meta-llama/llama-4-scout-17b-16e-instruct   (newer, multimodal - recommended)
#   llama-3.2-90b-vision-preview                (legacy)
#   llama-3.2-11b-vision-preview                (smaller, faster)
GROQ_VISION_MODEL = os.environ.get(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)


def _img_to_b64(image: Image.Image, max_side: int = 768) -> str:
    """Encode a PIL image as a base64 JPEG, resizing if too large."""
    img = image.convert("RGB")
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class GroqVisionModel:
    """MedGemmaModel-compatible wrapper around Groq's vision-language API."""

    def __init__(
        self,
        model_id: str = GROQ_VISION_MODEL,
        api_key: Optional[str] = None,
        load_in_4bit: bool = False,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get one free at https://console.groq.com "
                "and set it: $env:GROQ_API_KEY = 'gsk_...'"
            )
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = "groq-remote"
        self.processor = "groq-remote"

    def load(self) -> None:
        """No local model to load."""
        print(f"Groq vision backend ready: {self.model_id} (no local load) OK")

    def _call(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        b64 = _img_to_b64(image)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  Groq error: {e}")
            return ""

    def generate_report(self, image, prompt=None, max_new_tokens=450):
        prompt = prompt or REPORT_GEN_PROMPT
        text = self._call(image, prompt, max_new_tokens)
        return _enforce_report_structure(text)

    def answer_question(self, image, question, context="", max_new_tokens=250):
        prompt = QA_ANSWER_PROMPT_TEMPLATE.format(
            context=context if context else "No additional context available.",
            question=question,
        )
        return self._call(image, prompt, max_new_tokens)

    def batch_generate_reports(self, images, verbose=True):
        out = []
        for i, img in enumerate(images):
            if verbose:
                print(f"  [{i+1}/{len(images)}] Generating via Groq ...", end="\r")
            out.append(self.generate_report(img))
        if verbose:
            print(f"  Generated {len(out)} reports OK")
        return out


def _enforce_report_structure(text: str) -> str:
    """Ensure FINDINGS / IMPRESSION sections exist."""
    if not text:
        return ("FINDINGS:\n[no output]\n\n"
                "IMPRESSION:\nPlease refer to the findings above.")
    upper = text.upper()
    has_f = "FINDINGS" in upper
    has_i = "IMPRESSION" in upper
    if has_f and has_i:
        return text
    out = "FINDINGS:\n" + text.strip()
    if not has_i:
        out += "\n\nIMPRESSION:\nPlease refer to the findings above."
    return out
