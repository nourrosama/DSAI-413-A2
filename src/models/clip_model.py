"""
clip_model.py — OpenAI CLIP wrapper for image and text encoding.

Model: openai/clip-vit-base-patch32 (via HuggingFace Transformers)

Role in this system
-------------------
  Comparison baseline:
    - Mode 1: Encode X-rays → compare visual embeddings vs MedGemma's perception.
    - Mode 2: Alternative retrieval encoder (general-domain vs ColPali's medical focus).

  CLIP is NOT a generative model — it produces embeddings only.
  For Mode 1 generation comparison, CLIP embeddings are compared via cosine similarity
  against ground-truth report embeddings to measure image-text alignment.
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.config import CLIP_MODEL_ID


class CLIPEncoder:
    """
    Wrapper around CLIP for computing image and text embeddings.

    Usage
    -----
    >>> enc = CLIPEncoder()
    >>> enc.load()
    >>> img_embs = enc.embed_images(pil_images)   # (N, 512)
    >>> txt_embs = enc.embed_texts(["text..."])    # (N, 512)
    >>> scores   = enc.image_text_similarity(img_embs, txt_embs)  # (N,)
    """

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        print(f"Loading CLIP from '{self.model_id}' …")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        print("CLIP loaded ✓")

    def _check_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

    # ── Embedding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute L2-normalized CLIP image embeddings.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), float32.
        """
        self._check_loaded()
        all_embs = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            embs = self.model.get_image_features(**inputs)  # (B, D)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())

        return np.vstack(all_embs).astype(np.float32)

    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute L2-normalized CLIP text embeddings.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), float32.
        """
        self._check_loaded()
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            embs = self.model.get_text_features(**inputs)  # (B, D)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())

        return np.vstack(all_embs).astype(np.float32)

    @torch.no_grad()
    def embed_single_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image. Returns shape (1, D)."""
        return self.embed_images([image])

    # ── Similarity ────────────────────────────────────────────────────────────

    def image_text_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between image and text embeddings.

        Both arrays must already be L2-normalized (embed_images / embed_texts do this).

        Parameters
        ----------
        image_embeddings : (N, D)
        text_embeddings  : (N, D) or (M, D)

        Returns
        -------
        np.ndarray of shape (N, M) or (N,) if shapes match.
        """
        # Dot product of normalized vectors = cosine similarity
        sims = image_embeddings @ text_embeddings.T
        if sims.shape[0] == sims.shape[1]:
            # Return diagonal if it's a paired comparison
            return np.diag(sims)
        return sims

    def zero_shot_classify(
        self,
        image: Image.Image,
        candidate_labels: List[str],
    ) -> dict:
        """
        Zero-shot classify an X-ray against a list of clinical label strings.

        Parameters
        ----------
        candidate_labels : e.g. ["normal chest X-ray",
                                  "pneumonia with consolidation",
                                  "pleural effusion", ...]

        Returns
        -------
        dict  {label: probability} sorted by probability descending.
        """
        self._check_loaded()
        img_emb = self.embed_single_image(image)               # (1, D)
        txt_embs = self.embed_texts(candidate_labels)          # (N, D)

        # Cosine similarities → softmax probabilities
        logits = (img_emb @ txt_embs.T).squeeze(0)            # (N,)
        logits_tensor = torch.from_numpy(logits) * 100.0       # scale as CLIP does
        probs = torch.softmax(logits_tensor, dim=0).numpy()

        return dict(sorted(zip(candidate_labels, probs.tolist()), key=lambda x: -x[1]))
