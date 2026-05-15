"""
qa_dataset_creation.py — Build the QA dataset from MIMIC-CXR reports.

Strategy: Direct Groq generation
  For each report, send it to Groq with a single prompt asking for QA pairs
  in JSON format. Parse the response and save.

  That's it — no regex, no templates, no post-processing pipeline.

Output format (each record):
  {
    "id": "qa_00001",
    "image_path": "path/to/xray.png",
    "question": "Is there any pleural effusion?",
    "answer": "Yes, there is a small left pleural effusion.",
    "category": "presence",
    "source_report": "..."
  }
"""

import json
import os
import time
from typing import List, Dict, Optional

from tqdm import tqdm

from src.config import (
    QA_PAIRS_PATH,
    GROQ_API_KEY,
    GROQ_MODEL,
    QA_PAIRS_PER_REPORT,
    QA_TOTAL_TARGET,
)

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a radiologist creating a question-answer dataset from chest X-ray reports. "
    "Generate concise, clinically meaningful QA pairs directly from the report text. "
    "Every answer must be grounded in the report — do not add information not present in it."
)

USER_PROMPT_TEMPLATE = """\
Here is a radiology report from a chest X-ray:

\"\"\"
{report}
\"\"\"

Generate exactly {n} question-answer pairs based on this report.
Cover a mix of: presence/absence of findings, locations, and severity where mentioned.

Respond with ONLY a JSON array — no explanation, no markdown, no extra text.
Format:
[
  {{
    "question": "...",
    "answer": "...",
    "category": "presence" | "location" | "severity"
  }},
  ...
]"""


# ─── Core function ────────────────────────────────────────────────────────────

def generate_qa_for_report(
    report: str,
    client,
    model: str = GROQ_MODEL,
    n_pairs: int = QA_PAIRS_PER_REPORT,
) -> List[Dict]:
    """
    Call Groq once for a single report and return parsed QA pairs.
    Returns an empty list if the call fails or the response is malformed.
    """
    prompt = USER_PROMPT_TEMPLATE.format(report=report.strip(), n=n_pairs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        pairs = json.loads(raw)

        # Basic validation — keep only well-formed entries
        valid = []
        for p in pairs:
            if isinstance(p, dict) and "question" in p and "answer" in p:
                valid.append({
                    "question": str(p["question"]).strip(),
                    "answer":   str(p["answer"]).strip(),
                    "category": str(p.get("category", "general")).strip(),
                })
        return valid

    except json.JSONDecodeError:
        print(f"  ⚠ JSON parse error — skipping this report.")
        return []
    except Exception as e:
        print(f"  ⚠ Groq error: {e}")
        return []


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_qa_dataset(
    reports: List[str],
    image_paths: List[str],
    api_key: Optional[str] = None,
    model: str = GROQ_MODEL,
    n_pairs_per_report: int = QA_PAIRS_PER_REPORT,
    target_size: int = QA_TOTAL_TARGET,
    output_path: str = QA_PAIRS_PATH,
    delay: float = 0.5,
) -> List[Dict]:
    """
    Build the full QA dataset by calling Groq once per report.

    Parameters
    ----------
    reports       : List of ground-truth report strings from MIMIC-CXR.
    image_paths   : Parallel list of image file paths.
    api_key       : Groq API key (falls back to config / env var).
    n_pairs_per_report : How many QA pairs to request per report.
    target_size   : Stop once this many pairs are collected.
    output_path   : Where to save the final JSON file.
    delay         : Seconds to wait between API calls (rate limiting).

    Returns
    -------
    List of QA dicts.
    """
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Set it as an environment variable or pass api_key= explicitly."
        )

    from groq import Groq
    client = Groq(api_key=key)

    all_pairs = []

    for report, img_path in tqdm(
        zip(reports, image_paths),
        total=len(reports),
        desc="Generating QA pairs",
    ):
        if len(all_pairs) >= target_size:
            break

        pairs = generate_qa_for_report(report, client, model=model, n_pairs=n_pairs_per_report)

        for pair in pairs:
            pair["image_path"]    = img_path
            pair["source_report"] = report
            all_pairs.append(pair)

        time.sleep(delay)

    # Trim to target and assign IDs
    all_pairs = all_pairs[:target_size]
    for i, pair in enumerate(all_pairs):
        pair["id"] = f"qa_{i+1:05d}"

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\n✓ Saved {len(all_pairs)} QA pairs → {output_path}")
    _print_stats(all_pairs)
    return all_pairs


# ─── Stats ────────────────────────────────────────────────────────────────────

def _print_stats(pairs: List[Dict]) -> None:
    from collections import Counter
    cats = Counter(p.get("category", "general") for p in pairs)
    print("\nQA Dataset Statistics:")
    print(f"  Total pairs : {len(pairs)}")
    print(f"  Categories  : {dict(cats)}")


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.config import MIMIC_CSV_PATH, MIMIC_IMG_COL, MIMIC_TEXT_COL, MIMIC_IMG_BASE, DATASET_SUBSET_SIZE
    from src.preprocessing import load_mimic_subset

    parser = argparse.ArgumentParser(description="Build QA dataset from MIMIC-CXR reports via Groq.")
    parser.add_argument("--csv",    default=MIMIC_CSV_PATH)
    parser.add_argument("--subset", type=int, default=DATASET_SUBSET_SIZE)
    parser.add_argument("--pairs",  type=int, default=QA_PAIRS_PER_REPORT, help="QA pairs per report")
    parser.add_argument("--target", type=int, default=QA_TOTAL_TARGET,     help="Total pairs to collect")
    args = parser.parse_args()

    _, reports, paths = load_mimic_subset(
        args.csv,
        img_col=MIMIC_IMG_COL,
        text_col=MIMIC_TEXT_COL,
        subset_size=args.subset,
        image_base_dir=MIMIC_IMG_BASE,
    )

    build_qa_dataset(
        reports=reports,
        image_paths=paths,
        n_pairs_per_report=args.pairs,
        target_size=args.target,
    )
