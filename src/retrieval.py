"""
retrieval.py — FAISS index build and query logic.

Supports three retrieval backends (configured in config.py):
  - "colpali" : visual embeddings from ColPali (primary, mandatory)
  - "clip"    : visual embeddings from CLIP (comparison baseline)
  - "sbert"   : text embeddings from SentenceTransformers (text-side retrieval)

Index type: IndexFlatIP (dot product / cosine similarity on normalized vectors).
This is consistent with the A1 ColPali + FAISS setup.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from PIL import Image

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    FAISS_TOP_K,
    RETRIEVAL_BACKEND,
    SBERT_MODEL_ID,
)


# ─── Index builder ────────────────────────────────────────────────────────────

class RetrievalIndex:
    """
    Builds and queries a FAISS flat inner-product index.

    The index stores one embedding vector per document.
    Metadata (image path, report text, etc.) is stored in a parallel JSON list.

    Usage
    -----
    Build (offline, done once):
    >>> idx = RetrievalIndex()
    >>> idx.build(embeddings, metadata_list)
    >>> idx.save()

    Query (online, per request):
    >>> idx = RetrievalIndex.load()
    >>> results = idx.query(query_embedding, top_k=5)
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dim: Optional[int] = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Populate the index with pre-computed embeddings.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D), float32, L2-normalized.
        metadata   : List of N dicts, one per document.
                     Each dict should contain at least {"image_path": ..., "report": ...}.
        """
        assert len(embeddings) == len(metadata), "Embeddings and metadata must have the same length."
        assert embeddings.dtype == np.float32, "Embeddings must be float32."

        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # inner product = cosine on normalized vecs
        self.index.add(embeddings)
        self.metadata = metadata
        print(f"FAISS index built: {self.index.ntotal} vectors of dimension {self.embedding_dim}.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        index_path: str = FAISS_INDEX_PATH,
        meta_path: str = FAISS_META_PATH,
    ) -> None:
        """Save the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Index saved → {index_path}")
        print(f"Metadata saved → {meta_path}")

    @classmethod
    def load(
        cls,
        index_path: str = FAISS_INDEX_PATH,
        meta_path: str = FAISS_META_PATH,
    ) -> "RetrievalIndex":
        """Load a previously saved index from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No FAISS index found at {index_path}. Run build_index first.")
        obj = cls()
        obj.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            obj.metadata = json.load(f)
        obj.embedding_dim = obj.index.d
        print(f"Index loaded: {obj.index.ntotal} vectors of dimension {obj.embedding_dim}.")
        return obj

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = FAISS_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K most similar documents.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (1, D) or (D,), float32, L2-normalized.
        top_k           : Number of results to return.

        Returns
        -------
        List of dicts, each containing the metadata + a "score" key.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")

        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results


# ─── High-level pipeline helpers ──────────────────────────────────────────────

def build_index_from_images(
    images: List[Image.Image],
    reports: List[str],
    image_paths: List[str],
    backend: str = RETRIEVAL_BACKEND,
    save: bool = True,
    model=None,
) -> RetrievalIndex:
    """
    Full pipeline: embed images → build FAISS index → optionally save.

    Parameters
    ----------
    images      : List of PIL Images.
    reports     : Paired ground-truth reports (stored as metadata).
    image_paths : Paired image paths (stored as metadata).
    backend     : "colpali" | "clip" | "sbert"
    save        : Whether to persist the index to disk.
    model       : Optional pre-loaded model instance to reuse (avoids reloading).
    """
    embeddings = _embed_images(images, backend, model=model)

    metadata = [
        {"image_path": p, "report": r, "index": i}
        for i, (p, r) in enumerate(zip(image_paths, reports))
    ]

    idx = RetrievalIndex()
    idx.build(embeddings, metadata)
    if save:
        idx.save()
    return idx


def retrieve_for_query(
    query_image: Image.Image,
    index: RetrievalIndex,
    backend: str = RETRIEVAL_BACKEND,
    top_k: int = FAISS_TOP_K,
    model=None,
) -> List[Dict[str, Any]]:
    """
    Embed a query image and retrieve top-K similar documents from the index.

    Parameters
    ----------
    model : Optional pre-loaded model instance to reuse (avoids reloading).

    Returns
    -------
    List of result dicts with keys: image_path, report, score.
    """
    q_emb = _embed_query(query_image, backend, model=model)
    return index.query(q_emb, top_k=top_k)


def format_retrieved_context(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved results into a context string for the LLM prompt.

    Returns a numbered list of report snippets with similarity scores.
    """
    if not results:
        return "No relevant context retrieved."

    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        report = r.get("report", "")
        # Truncate very long reports for the prompt
        snippet = report[:500] + "…" if len(report) > 500 else report
        lines.append(f"[{i}] (similarity: {score:.3f})\n{snippet}")

    return "\n\n".join(lines)


# ─── Evaluation: Precision@K ──────────────────────────────────────────────────

def precision_at_k(
    query_image: Image.Image,
    ground_truth_path: str,
    index: RetrievalIndex,
    backend: str = RETRIEVAL_BACKEND,
    k: int = FAISS_TOP_K,
) -> float:
    """
    Compute Precision@K for a single query.
    A retrieved result is "correct" if its image_path matches ground_truth_path.
    """
    results = retrieve_for_query(query_image, index, backend=backend, top_k=k)
    relevant = sum(1 for r in results if r.get("image_path") == ground_truth_path)
    return relevant / k


# ─── Internal embedding dispatch ──────────────────────────────────────────────

def _embed_images(images: List[Image.Image], backend: str, model=None) -> np.ndarray:
    """
    Embed a list of images using the specified backend.
    Pass `model` to reuse an already-loaded instance and avoid repeated loading.
    """
    if backend == "colpali":
        from src.models.colpali import ColPaliModel
        m = model if model is not None else ColPaliModel()
        if model is None:
            m.load()
        return m.embed_images(images)
    elif backend == "clip":
        from src.models.clip_model import CLIPEncoder
        m = model if model is not None else CLIPEncoder()
        if model is None:
            m.load()
        return m.embed_images(images)
    elif backend == "sbert":
        raise ValueError("SBERT backend embeds text, not images. Use build_index_from_texts() instead.")
    else:
        raise ValueError(f"Unknown retrieval backend: {backend}")


def _embed_query(image: Image.Image, backend: str, model=None) -> np.ndarray:
    """
    Embed a single query image.
    Pass `model` to reuse an already-loaded instance and avoid repeated loading.
    """
    if backend == "colpali":
        from src.models.colpali import ColPaliModel
        m = model if model is not None else ColPaliModel()
        if model is None:
            m.load()
        return m.embed_query_image(image)
    elif backend == "clip":
        from src.models.clip_model import CLIPEncoder
        m = model if model is not None else CLIPEncoder()
        if model is None:
            m.load()
        return m.embed_single_image(image)
    else:
        raise ValueError(f"Unknown retrieval backend for image query: {backend}")


def build_index_from_texts(
    texts: List[str],
    image_paths: List[str],
    save: bool = True,
    index_path: str = FAISS_INDEX_PATH,
    meta_path: str = FAISS_META_PATH,
) -> RetrievalIndex:
    """
    Alternative: build a FAISS index from text embeddings (report-side retrieval).
    Uses sentence-transformers.
    """
    from sentence_transformers import SentenceTransformer
    print(f"Loading SBERT '{SBERT_MODEL_ID}' …")
    sbert = SentenceTransformer(SBERT_MODEL_ID)
    embeddings = sbert.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    metadata = [{"image_path": p, "report": t, "index": i} for i, (p, t) in enumerate(zip(image_paths, texts))]
    idx = RetrievalIndex()
    idx.build(embeddings, metadata)
    if save:
        idx.save(index_path, meta_path)
    return idx
