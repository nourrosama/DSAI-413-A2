"""
mode2_qa.py — Mode 2: QA Pipeline (RAG-Based)

Input  : A chest X-ray image + a natural-language clinical question
Output : A grounded clinical answer + retrieved context + similarity scores

Pipeline
--------
1. Load and preprocess the query image.
2. Encode the image using ColPali (or CLIP as alternative).
3. Query the FAISS index → retrieve top-K similar X-ray reports.
4. Format retrieved reports as context.
5. Construct a prompt: image + question + context.
6. Generate the answer using MedGemma.
7. Return answer, retrieved context, and metadata.
"""

from typing import Dict, List, Optional

from PIL import Image

from src.config import FAISS_TOP_K, RETRIEVAL_BACKEND
from src.preprocessing import load_image
from src.retrieval import RetrievalIndex, retrieve_for_query, format_retrieved_context


# ─── Main pipeline class ──────────────────────────────────────────────────────

class QAPipeline:
    """
    End-to-end Mode 2 RAG pipeline.

    Usage
    -----
    # Build the index first (offline, once):
    >>> from src.retrieval import build_index_from_images
    >>> index = build_index_from_images(images, reports, paths)

    # Then run QA:
    >>> pipeline = QAPipeline(index=index)
    >>> pipeline.load_models()
    >>> result = pipeline.run(image, "Is there any evidence of pneumonia?")
    >>> print(result["answer"])
    """

    def __init__(
        self,
        index: Optional[RetrievalIndex] = None,
        retrieval_backend: str = RETRIEVAL_BACKEND,
        top_k: int = FAISS_TOP_K,
        medgemma_load_in_4bit: bool = True,
    ):
        self.index = index
        self.retrieval_backend = retrieval_backend
        self.top_k = top_k
        self.medgemma_load_in_4bit = medgemma_load_in_4bit

        self._medgemma = None
        self._colpali = None
        self._clip = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load MedGemma and the retrieval encoder. Call once before inference."""
        # Always load MedGemma for answer generation
        from src.models.medgemma import MedGemmaModel
        self._medgemma = MedGemmaModel(load_in_4bit=self.medgemma_load_in_4bit)
        self._medgemma.load()

        # Load the retrieval encoder
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

    # ── Internal: embed query image ───────────────────────────────────────────

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

    # ── Single query inference ────────────────────────────────────────────────

    def run(
        self,
        image: Optional[Image.Image] = None,
        question: str = "",
        image_path: Optional[str] = None,
        ground_truth_answer: Optional[str] = None,
    ) -> Dict:
        """
        Run the full Mode 2 pipeline.

        Parameters
        ----------
        image                : PIL Image (if already loaded).
        question             : Natural-language clinical question.
        image_path           : Path to image file (used if image is None).
        ground_truth_answer  : Reference answer for metric computation (optional).

        Returns
        -------
        dict with keys:
          "answer"            : Generated clinical answer (str)
          "question"          : The input question (str)
          "retrieved_context" : Formatted context string used in the prompt (str)
          "retrieved_results" : Raw list of retrieval result dicts
          "retrieval_backend" : Which encoder was used
          "metrics"           : QA accuracy metrics if ground_truth_answer provided
        """
        if not question:
            raise ValueError("A question must be provided for Mode 2.")

        # 1. Load image
        if image is None and image_path is None:
            raise ValueError("Provide either `image` or `image_path`.")
        if image is None:
            image = load_image(image_path)

        # 2. Retrieve similar cases
        if self.index is None:
            raise RuntimeError("No FAISS index loaded. Call load_index() or pass index= in constructor.")

        q_emb = self._embed_query(image)
        retrieved = self.index.query(q_emb, top_k=self.top_k)
        context_str = format_retrieved_context(retrieved)

        # 3. Generate answer with MedGemma
        if self._medgemma is None:
            raise RuntimeError("MedGemma not loaded. Call load_models() first.")

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
            "metrics": None,
        }

        # 4. Optional evaluation
        if ground_truth_answer and answer:
            from src.evaluation import compute_qa_metrics
            result["metrics"] = compute_qa_metrics([answer], [ground_truth_answer])

        return result

    # ── Batch inference ───────────────────────────────────────────────────────

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
                print(f"  [{i+1}/{len(images)}] {q[:60]}…", end="\r")
            gt = ground_truth_answers[i] if ground_truth_answers else None
            results.append(self.run(image=img, question=q, ground_truth_answer=gt))

        if verbose:
            print(f"\n  Processed {len(results)} QA pairs.")
        return results

    # ── Retrieval comparison ──────────────────────────────────────────────────

    def compare_backends(
        self,
        image: Image.Image,
        question: str,
        backends: List[str] = ("colpali", "clip"),
    ) -> Dict[str, Dict]:
        """
        Run the same query with multiple retrieval backends and compare results.
        Useful for the model comparison section.

        Note: This reloads each encoder in sequence — intended for evaluation, not production.
        """
        comparison = {}
        original_backend = self.retrieval_backend

        for backend in backends:
            print(f"  Testing backend: {backend} …")
            self.retrieval_backend = backend

            # Reload the appropriate encoder
            if backend == "colpali":
                from src.models.colpali import ColPaliModel
                self._colpali = ColPaliModel()
                self._colpali.load()
            elif backend == "clip":
                from src.models.clip_model import CLIPEncoder
                self._clip = CLIPEncoder()
                self._clip.load()

            comparison[backend] = self.run(image=image, question=question)

        self.retrieval_backend = original_backend
        return comparison
