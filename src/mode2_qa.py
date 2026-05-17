"""
mode2_qa.py - Mode 2: QA Pipeline (RAG-Based)

Input  : A chest X-ray image + a natural-language clinical question
Output : A grounded clinical answer + retrieved context + similarity scores

Backend selection (generator)
-----------------------------
Picked at .load_models() time based on env var USE_GROQ_VISION:
  USE_GROQ_VISION=1  -> Groq cloud (Llama 4 Scout vision) - no GPU required
  unset / 0          -> MedGemma 4B local (requires CUDA GPU + bf16)

Retrieval backend (ColPali vs CLIP) is controlled via constructor argument.

Pipeline
--------
1. Load and preprocess the query image.
2. Encode the image using ColPali (or CLIP as alternative).
3. Query the FAISS index -> retrieve top-K similar X-ray reports.
4. Format retrieved reports as context.
5. Construct a prompt: image + question + context.
6. Generate the answer using the selected backend.
7. Return answer, retrieved context, and metadata.
"""

import os
from typing import Dict, List, Optional

from PIL import Image

from src.config import FAISS_TOP_K, RETRIEVAL_BACKEND
from src.preprocessing import load_image
from src.retrieval import RetrievalIndex, format_retrieved_context


def _use_groq() -> bool:
    """True if env var USE_GROQ_VISION is set to "1" / "true" / "yes"."""
    return os.environ.get("USE_GROQ_VISION", "0").lower() in ("1", "true", "yes")


class QAPipeline:
    """End-to-end Mode 2 RAG pipeline."""

    def __init__(
        self,
        index: Optional[RetrievalIndex] = None,
        retrieval_backend: str = RETRIEVAL_BACKEND,
        top_k: int = FAISS_TOP_K,
        medgemma_load_in_4bit: bool = False,
    ):
        self.index = index
        self.retrieval_backend = retrieval_backend
        self.top_k = top_k
        self.medgemma_load_in_4bit = medgemma_load_in_4bit

        self._medgemma = None        # MedGemma or Groq instance
        self._colpali = None
        self._clip = None
        self._backend = "unset"

    # -- Model loading ---------------------------------------------------------

    def load_models(self) -> None:
        """Load the generator and the retrieval encoder."""
        # 1. Generator (MedGemma or Groq)
        if _use_groq():
            from src.models.groq_vision import GroqVisionModel
            self._medgemma = GroqVisionModel()
            self._backend = "groq"
        else:
            from src.models.medgemma import MedGemmaModel
            self._medgemma = MedGemmaModel(load_in_4bit=self.medgemma_load_in_4bit)
            self._backend = "medgemma"
        self._medgemma.load()
        print(f"Mode 2 generator backend: {self._backend}")

        # 2. Retrieval encoder (ColPali or CLIP)
        if self.retrieval_backend == "colpali":
            from src.models.colpali import ColPaliModel
            self._colpali = ColPaliModel()
            self._colpali.load()
        elif self.retrieval_backend == "clip":
            from src.models.clip_model import CLIPEncoder
            self._clip = CLIPEncoder()
            self._clip.load()

    def load_index(
        self,
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
    ) -> None:
        """Load a pre-built FAISS index from disk."""
        from src.config import FAISS_INDEX_PATH, FAISS_META_PATH
        self.index = RetrievalIndex.load(
            index_path or FAISS_INDEX_PATH,
            meta_path or FAISS_META_PATH,
        )

    # -- Internal: embed query image -------------------------------------------

    def _embed_query(self, image: Image.Image):
        """Return (1, D) embedding for the query image using the active backend."""
        if self.retrieval_backend == "colpali":
            if self._colpali is None:
                raise RuntimeError("ColPali not loaded.")
            return self._colpali.embed_query_image(image)
        elif self.retrieval_backend == "clip":
            if self._clip is None:
                raise RuntimeError("CLIP not loaded.")
            return self._clip.embed_single_image(image)
        else:
            raise ValueError(f"Unknown retrieval backend: {self.retrieval_backend}")

    # -- Single query inference ------------------------------------------------

    def run(
        self,
        image: Optional[Image.Image] = None,
        question: str = "",
        image_path: Optional[str] = None,
        ground_truth_answer: Optional[str] = None,
    ) -> Dict:
        """
        Run the full Mode 2 pipeline.

        Returns
        -------
        dict with keys: answer, question, retrieved_context, retrieved_results,
                       retrieval_backend, generator_backend, metrics.
        """
        if not question:
            raise ValueError("A question must be provided for Mode 2.")

        if image is None and image_path is None:
            raise ValueError("Provide either `image` or `image_path`.")
        if image is None:
            image = load_image(image_path)

        if self.index is None:
            raise RuntimeError("No FAISS index loaded. Call load_index() or pass index= in constructor.")

        # Retrieve similar cases
        q_emb = self._embed_query(image)
        retrieved = self.index.query(q_emb, top_k=self.top_k)
        context_str = format_retrieved_context(retrieved)

        # Generate answer
        if self._medgemma is None:
            raise RuntimeError("Generator not loaded. Call load_models() first.")
        answer = self._medgemma.answer_question(
            image=image,
            question=question,
            context=context_str,
        )

        result = {
            "answer": answer,
            "question": question,
            "retrieved_context": context_str,
            "retrieved_results": retrieved,
            "retrieval_backend": self.retrieval_backend,
            "generator_backend": self._backend,
            "metrics": None,
        }

        if ground_truth_answer and answer:
            from src.evaluation import compute_qa_metrics
            result["metrics"] = compute_qa_metrics([answer], [ground_truth_answer])

        return result

    # -- Batch inference -------------------------------------------------------

    def run_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        ground_truth_answers: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """Run the pipeline over a list of (image, question) pairs."""
        assert len(images) == len(questions), "images and questions must have equal length."
        results = []
        for i, (img, q) in enumerate(zip(images, questions)):
            if verbose:
                print(f"  [{i+1}/{len(images)}] {q[:60]}...", end="\r")
            gt = ground_truth_answers[i] if ground_truth_answers else None
            results.append(self.run(image=img, question=q, ground_truth_answer=gt))
        if verbose:
            print(f"\n  Processed {len(results)} QA pairs.")
        return results
