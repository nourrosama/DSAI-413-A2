"""
evaluation.py — Evaluation metrics for both modes and retrieval.

Metrics
-------
Mode 1 (Report Generation):
  - BLEU-1, BLEU-4       (nltk)
  - ROUGE-L              (rouge-score)
  - BERTScore F1         (bert-score)

Mode 2 (QA):
  - Exact Match (EM)
  - Token-level F1       (same as SQuAD evaluation)

Retrieval (ColPali / CLIP):
  - Precision@K

Comparison table:
  - generate_comparison_table() summarizes all models side-by-side.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Report generation metrics ───────────────────────────────────────────────

def compute_report_metrics(
    hypotheses: List[str],
    references: List[str],
    use_bertscore: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU, ROUGE-L, and optionally BERTScore for a list of generated
    reports vs ground-truth references.

    Parameters
    ----------
    hypotheses : List of generated report strings.
    references : List of ground-truth report strings (parallel).

    Returns
    -------
    dict with keys: bleu1, bleu4, rouge_l, bertscore_f1 (if enabled).
    """
    assert len(hypotheses) == len(references), "hypotheses and references must have equal length."

    results = {}

    # ── BLEU ──────────────────────────────────────────────────────────────────
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        tokenize = nltk.word_tokenize
        smooth = SmoothingFunction().method1

        ref_tokens  = [[tokenize(r.lower())] for r in references]
        hyp_tokens  = [tokenize(h.lower()) for h in hypotheses]

        results["bleu1"] = round(
            corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth), 4
        )
        results["bleu4"] = round(
            corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth), 4
        )
    except Exception as e:
        print(f"  BLEU computation failed: {e}")
        results["bleu1"] = None
        results["bleu4"] = None

    # ── ROUGE-L ───────────────────────────────────────────────────────────────
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
        results["rouge_l"] = round(float(np.mean(rouge_scores)), 4)
    except Exception as e:
        print(f"  ROUGE-L computation failed: {e}")
        results["rouge_l"] = None

    # ── BERTScore ─────────────────────────────────────────────────────────────
    if use_bertscore:
        try:
            from bert_score import score as bert_score
            from src.config import BERTSCORE_LANG
            P, R, F = bert_score(hypotheses, references, lang=BERTSCORE_LANG, verbose=False)
            results["bertscore_f1"] = round(float(F.mean()), 4)
        except Exception as e:
            print(f"  BERTScore computation failed: {e}")
            results["bertscore_f1"] = None

    return results


# ─── QA metrics ───────────────────────────────────────────────────────────────

def compute_qa_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Compute SQuAD-style Exact Match and Token F1 for QA predictions.

    Parameters
    ----------
    predictions  : List of model-generated answers.
    ground_truths: List of reference answers (parallel).

    Returns
    -------
    dict with keys: exact_match, token_f1.
    """
    assert len(predictions) == len(ground_truths)

    em_scores = []
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_norm = _normalize_answer(pred)
        gt_norm   = _normalize_answer(gt)
        em_scores.append(float(pred_norm == gt_norm))
        f1_scores.append(_token_f1(pred_norm, gt_norm))

    return {
        "exact_match": round(float(np.mean(em_scores)), 4),
        "token_f1":    round(float(np.mean(f1_scores)), 4),
        "n_samples":   len(predictions),
    }


def _normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def _token_f1(pred: str, gt: str) -> float:
    """Compute token-level F1 between two normalized strings."""
    pred_tokens = pred.split()
    gt_tokens   = gt.split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall    = num_same / len(gt_tokens)   if gt_tokens   else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


# ─── Retrieval evaluation: Precision@K ───────────────────────────────────────

def compute_precision_at_k(
    retrieved_results_list: List[List[Dict]],
    relevant_paths_list: List[List[str]],
    k: int = 5,
) -> float:
    """
    Compute mean Precision@K over a set of queries.

    Parameters
    ----------
    retrieved_results_list : List of retrieval result lists (one per query).
    relevant_paths_list    : For each query, the list of ground-truth relevant image paths.
    k                      : @K cutoff.

    Returns
    -------
    Mean Precision@K (float).
    """
    precisions = []
    for results, relevant in zip(retrieved_results_list, relevant_paths_list):
        top_k = results[:k]
        relevant_set = set(relevant)
        hits = sum(1 for r in top_k if r.get("image_path") in relevant_set)
        precisions.append(hits / k)
    return round(float(np.mean(precisions)), 4)


# ─── Comparison table ─────────────────────────────────────────────────────────

def generate_comparison_table(
    model_results: Dict[str, Dict],
) -> str:
    """
    Generate a markdown comparison table from a dict of model evaluation results.

    Parameters
    ----------
    model_results : {
        "MedGemma": {"task": "Report Gen", "bleu1": ..., "rouge_l": ..., "bertscore_f1": ...},
        "ColPali":  {"task": "Retrieval",  "precision_at_k": ...},
        "CLIP":     {"task": "Retrieval",  "precision_at_k": ...},
    }

    Returns
    -------
    Markdown table string.
    """
    headers = ["Model", "Task", "BLEU-1", "BLEU-4", "ROUGE-L", "BERTScore F1", "EM", "Token F1", "P@K"]
    rows = []

    for model_name, metrics in model_results.items():
        row = [
            model_name,
            metrics.get("task", "—"),
            _fmt(metrics.get("bleu1")),
            _fmt(metrics.get("bleu4")),
            _fmt(metrics.get("rouge_l")),
            _fmt(metrics.get("bertscore_f1")),
            _fmt(metrics.get("exact_match")),
            _fmt(metrics.get("token_f1")),
            _fmt(metrics.get("precision_at_k")),
        ]
        rows.append(row)

    # Build markdown
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    data_rows = [
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]

    return "\n".join([header_row, sep] + data_rows)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# ─── Per-example evaluation (for qualitative analysis) ───────────────────────

def evaluate_single_report(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute all metrics for a single report pair. Useful in notebooks."""
    return compute_report_metrics([hypothesis], [reference])


def evaluate_single_qa(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute EM and F1 for a single QA pair. Useful in notebooks."""
    return compute_qa_metrics([prediction], [ground_truth])
