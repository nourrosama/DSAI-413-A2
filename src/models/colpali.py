"""
colpali.py — ColPali loader and embedding wrapper.

ColPali uses a PaliGemma backbone with a late-interaction (MaxSim / ColBERT-style)
scoring mechanism. It produces per-patch token embeddings for images, making it
ideal for visual retrieval rather than generation.

Model: vidore/colpali-v1.2

Role in this system
-------------------
  Mode 2 (QA / RAG): Encode X-ray images → FAISS index → retrieve similar cases.
  Mode 1 (optional): Can be used as a visual baseline encoder for comparison.
"""

from typing import List, Optional

import torch
import numpy as np
from PIL import Image

# ColPali engine provides clean wrappers around the raw HuggingFace model
from colpali_engine.models import ColPali, ColPaliProcessor

from src.config import COLPALI_MODEL_ID, COLPALI_BATCH_SIZE


class ColPaliModel:
    """
    Wrapper around ColPali for computing image embeddings used in retrieval.

    The embedding returned by `embed_images` is a 2-D array of shape
    (N, embedding_dim), computed by mean-pooling the per-patch token embeddings.
    This allows FAISS IndexFlatIP dot-product search.

    Usage
    -----
    >>> model = ColPaliModel()
    >>> model.load()
    >>> embeddings = model.embed_images(pil_images)   # np.ndarray (N, D)
    >>> query_emb  = model.embed_query_image(single_pil_image)  # (1, D)
    """

    def __init__(
        self,
        model_id: str = COLPALI_MODEL_ID,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load ColPali model and processor. Call once before inference."""
        print(f"Loading ColPali from '{self.model_id}' …")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = ColPali.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
        ).eval()

        self.processor = ColPaliProcessor.from_pretrained(self.model_id)
        print("ColPali loaded ✓")

    def _check_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

    # ── Embedding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = COLPALI_BATCH_SIZE,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute mean-pooled ColPali embeddings for a list of PIL Images.

        Parameters
        ----------
        images     : List of PIL Images (RGB).
        batch_size : Number of images per forward pass.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), float32, L2-normalized.
        """
        self._check_loaded()
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            if verbose:
                print(f"  Embedding batch {i // batch_size + 1} / {len(images) // batch_size + 1} …", end="\r")

            inputs = self.processor.process_images(batch).to(self.model.device)
            # Shape: (batch, num_patches, embedding_dim)
            patch_embeddings = self.model(**inputs)

            # Mean-pool patches → one vector per image
            # patch_embeddings: (B, seq_len, D)
            mean_emb = patch_embeddings.mean(dim=1)  # (B, D)
            all_embeddings.append(mean_emb.cpu().float().numpy())

        if verbose:
            print()

        embeddings = np.vstack(all_embeddings)
        # L2-normalize for cosine / dot-product similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return (embeddings / norms).astype(np.float32)

    @torch.no_grad()
    def embed_query_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a single query image.

        Returns
        -------
        np.ndarray of shape (1, embedding_dim), L2-normalized.
        """
        return self.embed_images([image], verbose=False)

    @torch.no_grad()
    def embed_text_query(self, text: str) -> np.ndarray:
        """
        Embed a text query using ColPali's text encoder (for text-side retrieval).

        Returns
        -------
        np.ndarray of shape (1, embedding_dim), L2-normalized.
        """
        self._check_loaded()
        inputs = self.processor.process_queries([text]).to(self.model.device)
        # ColPali query embeddings: (1, seq_len, D)
        query_emb = self.model(**inputs)
        mean_emb = query_emb.mean(dim=1).cpu().float().numpy()
        norm = np.linalg.norm(mean_emb, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-9, norm)
        return (mean_emb / norm).astype(np.float32)

    # ── MaxSim scoring (late-interaction, optional) ───────────────────────────

    @torch.no_grad()
    def maxsim_score(
        self,
        query_image: Image.Image,
        corpus_images: List[Image.Image],
    ) -> np.ndarray:
        """
        Compute ColPali MaxSim scores between a query image and a corpus.
        More accurate than mean-pooling but slower — use for re-ranking.

        Returns
        -------
        np.ndarray of shape (len(corpus_images),) with similarity scores.
        """
        self._check_loaded()

        q_input = self.processor.process_images([query_image]).to(self.model.device)
        q_emb = self.model(**q_input)  # (1, seq_len_q, D)

        scores = []
        for img in corpus_images:
            d_input = self.processor.process_images([img]).to(self.model.device)
            d_emb = self.model(**d_input)  # (1, seq_len_d, D)
            # MaxSim: for each query token, find max similarity across document tokens
            # q_emb: (1, seq_len_q, D) → squeeze to (seq_len_q, D)
            # d_emb: (1, seq_len_d, D) → squeeze to (seq_len_d, D)
            q = q_emb.squeeze(0)  # (seq_len_q, D)
            d = d_emb.squeeze(0)  # (seq_len_d, D)
            sim = torch.einsum("qd,nd->qn", q, d)  # (seq_len_q, seq_len_d)
            score = sim.max(dim=-1).values.sum().item()
            scores.append(score)

        return np.array(scores, dtype=np.float32)
