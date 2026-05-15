"""
config.py — Central configuration for all paths, model names, and hyperparameters.
All other modules import from here. Change values here only.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR          = os.path.join(BASE_DIR, "data")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
QA_DATASET_DIR    = os.path.join(DATA_DIR, "qa_dataset")
QA_PAIRS_PATH     = os.path.join(QA_DATASET_DIR, "qa_pairs.json")

FAISS_INDEX_PATH  = os.path.join(DATA_DIR, "faiss_index.bin")
FAISS_META_PATH   = os.path.join(DATA_DIR, "faiss_metadata.json")

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Path to the MIMIC-CXR metadata CSV (set after Kaggle download)
MIMIC_CSV_PATH    = os.path.join(DATA_DIR, "mimic-cxr", "mimic_cxr_aug_train.csv")
# Base directory that image paths in the CSV are relative to
MIMIC_IMG_BASE    = os.path.join(DATA_DIR, "mimic-cxr", "official_data_iccv_final")
# Column names in the CSV
MIMIC_IMG_COL     = "image"        # contains a stringified list of image paths
MIMIC_TEXT_COL    = "text"         # contains a stringified list with the report

# Subset size for development (None = full dataset)
DATASET_SUBSET_SIZE = 1000

# Train / validation / test split ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# ─── Models ───────────────────────────────────────────────────────────────────
# MedGemma
MEDGEMMA_MODEL_ID  = "google/medgemma-4b-it"
MEDGEMMA_DTYPE     = "float16"          # use bfloat16 on Ampere+ GPUs
MEDGEMMA_MAX_TOKENS = 450               # max new tokens for report generation
MEDGEMMA_QA_MAX_TOKENS = 250            # max new tokens for QA answers

# ColPali
COLPALI_MODEL_ID   = "vidore/colpali-v1.2"
COLPALI_BATCH_SIZE = 4                  # images per batch during indexing

# CLIP
CLIP_MODEL_ID      = "openai/clip-vit-base-patch32"

# Sentence-Transformers (text retrieval fallback)
SBERT_MODEL_ID     = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Retrieval ────────────────────────────────────────────────────────────────
FAISS_TOP_K        = 5                  # number of documents to retrieve
RETRIEVAL_BACKEND  = "colpali"          # "colpali" | "clip" | "sbert"

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE   = 100               # number of test examples for eval
BERTSCORE_LANG     = "en"

# ─── QA Dataset Creation ──────────────────────────────────────────────────────
GROQ_MODEL         = "llama-3.3-70b-versatile"  # Groq model for QA generation
QA_PAIRS_PER_REPORT = 3                 # QA pairs to generate per report
QA_TOTAL_TARGET    = 500               # total pairs to aim for

# ─── Prompts ──────────────────────────────────────────────────────────────────
REPORT_GEN_PROMPT = (
    "You are an expert radiologist. Carefully analyze the chest X-ray image provided "
    "and generate a structured clinical radiology report. Your report must contain "
    "exactly two sections:\n\n"
    "FINDINGS:\n[Describe all visible findings in the lungs, heart, mediastinum, "
    "pleura, and bones. Be specific about location, size, and character.]\n\n"
    "IMPRESSION:\n[Summarize the key findings and provide a concise clinical interpretation.]\n\n"
    "Be precise, use standard radiological terminology, and avoid speculation beyond "
    "what is visible in the image."
)

QA_ANSWER_PROMPT_TEMPLATE = (
    "You are a clinical radiologist answering questions about chest X-rays.\n\n"
    "Retrieved Context (from similar cases):\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question based on the provided chest X-ray image and the retrieved "
    "context. Be concise and clinically precise. If the finding is not visible in "
    "the image, say so explicitly."
)

# ─── Groq ─────────────────────────────────────────────────────────────────────
# Set GROQ_API_KEY as an environment variable — do not hard-code here.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
