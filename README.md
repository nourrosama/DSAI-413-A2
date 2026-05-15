# Chest X-Ray Intelligence System
### DSAI 413 — Assignment 2 | Zewail City

A dual-mode, multi-modal medical AI system for chest X-ray analysis.

```
                        ┌─────────────────────────┐
                        │      Streamlit App        │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
          ┌─────────▼──────────┐            ┌─────────▼──────────┐
          │  MODE 1             │            │  MODE 2             │
          │  Report Generation  │            │  QA (RAG-based)     │
          │                     │            │                     │
          │  Image → MedGemma → │            │  Image + Question → │
          │  Structured Report  │            │  ColPali → FAISS →  │
          │                     │            │  MedGemma → Answer  │
          └─────────────────────┘            └─────────────────────┘
```

---

## Models

| Model | Role | Source |
|---|---|---|
| **MedGemma-4B-IT** | Report generation & QA answering | [HuggingFace](https://huggingface.co/google/medgemma-4b-it) |
| **ColPali v1.2** | Visual image retrieval (RAG) | [HuggingFace](https://huggingface.co/vidore/colpali-v1.2) |
| **CLIP ViT-B/32** | Retrieval baseline & alignment scoring | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- CUDA GPU (recommended — MedGemma 4B loaded in 4-bit quantization)
- HuggingFace account with MedGemma license accepted:
  → https://huggingface.co/google/medgemma-4b-it

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure paths

Edit `src/config.py`:
```python
MIMIC_CSV_PATH = "path/to/your/mimic_cxr_metadata.csv"
```

Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export GROQ_API_KEY="your_groq_api_key"   # for QA dataset creation
```

### 4. Download MIMIC-CXR

Dataset: https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset

---

## Usage

### Run notebooks (recommended order)

```
notebooks/01_data_exploration.ipynb   → Explore data, build train/val/test split
notebooks/02_report_generation.ipynb  → Mode 1 pipeline + evaluation
notebooks/03_qa_pipeline.ipynb        → Build QA dataset + FAISS index + Mode 2
notebooks/04_model_comparison.ipynb   → Full model comparison table
```

### Run the Streamlit demo

```bash
streamlit run app/app.py
```

For Google Colab:
```python
!streamlit run app/app.py &
from pyngrok import ngrok
print(ngrok.connect(8501))
```

### Build QA dataset from CLI

```bash
python -m src.qa_dataset_creation --subset 1000
```

---

## Repository Structure

```
chest-xray-intelligence/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sample_images/          # A few sample X-rays for demo
│   ├── qa_dataset/
│   │   └── qa_pairs.json       # Generated QA dataset
│   ├── faiss_index.bin         # ColPali FAISS index (built at runtime)
│   ├── faiss_metadata.json     # Index metadata
│   └── split_indices.json      # Train/val/test indices
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_report_generation.ipynb
│   ├── 03_qa_pipeline.ipynb
│   └── 04_model_comparison.ipynb
│
├── src/
│   ├── config.py               # All paths, model names, hyperparameters
│   ├── preprocessing.py        # Image loading (PNG/JPG/DICOM), normalization
│   ├── models/
│   │   ├── medgemma.py         # MedGemma inference wrapper
│   │   ├── colpali.py          # ColPali embedding wrapper
│   │   └── clip_model.py       # CLIP encoder wrapper
│   ├── mode1_report_gen.py     # Mode 1 pipeline
│   ├── mode2_qa.py             # Mode 2 RAG pipeline
│   ├── retrieval.py            # FAISS index build + query logic
│   ├── evaluation.py           # BLEU, ROUGE-L, BERTScore, EM, F1, P@K
│   └── qa_dataset_creation.py  # Hybrid template + Groq QA generation
│
└── app/
    └── app.py                  # Streamlit demo application
```

---

## Evaluation

### Mode 1 — Report Generation

| Metric | Description |
|---|---|
| BLEU-1 / BLEU-4 | N-gram overlap with ground-truth reports |
| ROUGE-L | Longest common subsequence overlap |
| BERTScore F1 | Semantic similarity using contextual embeddings |
| CLIP alignment | Cosine similarity between image and generated report embeddings |

### Mode 2 — QA

| Metric | Description |
|---|---|
| Exact Match (EM) | Proportion of predictions that exactly match the reference answer |
| Token F1 | Token-level overlap between prediction and reference |

### Retrieval

| Metric | Description |
|---|---|
| Precision@K | Fraction of top-K retrieved documents that are relevant |

---

## QA Dataset

The QA dataset is constructed from MIMIC-CXR ground-truth reports using a **hybrid approach**:

1. **Template extraction** — clinical facts (findings, locations, severities) are extracted using regex patterns and converted to question-answer pairs.
2. **Groq LLM augmentation** — questions are rephrased into natural clinical language using Llama-3 via the Groq API.
3. **Filtering and balancing** — duplicates removed; yes/no answers balanced.

Output: `data/qa_dataset/qa_pairs.json` — JSON with fields: `id`, `image_path`, `question`, `answer`, `category`, `source_report`, `generation_method`.

---

## Assignment Reference

- Build guide: `DSAI413_A2_Build_Guide.md`
- Reference repo: https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation
- ColPali paper / HF cookbook: https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms
