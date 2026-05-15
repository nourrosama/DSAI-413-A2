"""
app.py — Streamlit demo application for the Multi-Modal Chest X-Ray Intelligence System.

Two tabs:
  Tab 1 — Mode 1: Report Generation  (image → structured report)
  Tab 2 — Mode 2: QA (RAG-based)     (image + question → grounded answer)

Run:
    streamlit run app/app.py

Colab tunnel:
    !streamlit run app/app.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Path setup (so `src` imports work when run from project root) ─────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Intelligence System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .report-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        font-family: monospace;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    .answer-box {
        background-color: #f0fff4;
        border-left: 4px solid #28a745;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .context-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-family: monospace;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "mode1_pipeline" not in st.session_state:
    st.session_state.mode1_pipeline = None
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False


# ─── Sidebar — Model settings ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Chest_Xray_PA_3-8-2010.png/220px-Chest_Xray_PA_3-8-2010.png", width=200)
    st.markdown("## ⚙️ Settings")

    load_in_4bit = st.checkbox("Load MedGemma in 4-bit (saves VRAM)", value=True)
    retrieval_backend = st.selectbox("Retrieval backend", ["colpali", "clip"], index=0)
    top_k = st.slider("Retrieval top-K", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown("### 📌 About")
    st.markdown("""
    **DSAI 413 — Assignment 2**
    Multi-Modal Chest X-Ray Intelligence System

    **Models:**
    • MedGemma-4B-IT (generation)
    • ColPali v1.2 (retrieval)
    • CLIP ViT-B/32 (comparison)

    **Dataset:** MIMIC-CXR
    """)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🫁 Chest X-Ray Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Modal AI for Radiology Report Generation & Clinical QA</div>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Mode 1 — Report Generation", "💬 Mode 2 — Clinical QA (RAG)"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Report Generation
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload a Chest X-Ray → Get a Structured Radiology Report")
    st.caption("MedGemma analyzes the image and generates a FINDINGS + IMPRESSION report.")

    col_upload, col_result = st.columns([1, 1.4], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload X-ray image",
            type=["png", "jpg", "jpeg"],
            key="mode1_upload",
        )

        prompt_choice = st.selectbox(
            "Prompt variant",
            ["default", "brief", "detailed", "few_shot"],
            help="Experiment with different prompts and observe output differences.",
        )

        show_clip_score = st.checkbox("Show CLIP alignment score", value=True)
        ground_truth = st.text_area(
            "Ground-truth report (optional, for metric evaluation)",
            height=100,
            placeholder="Paste the reference report here to compute BLEU/ROUGE/BERTScore…",
        )

        generate_btn = st.button("🔍 Generate Report", type="primary", use_container_width=True)

    with col_result:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if generate_btn and uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            with st.spinner("Loading MedGemma and generating report…"):
                try:
                    # Lazy load pipeline
                    if st.session_state.mode1_pipeline is None:
                        from src.mode1_report_gen import ReportGenerationPipeline
                        pipeline = ReportGenerationPipeline(
                            use_clip=show_clip_score,
                            medgemma_load_in_4bit=load_in_4bit,
                        )
                        pipeline.load_models()
                        st.session_state.mode1_pipeline = pipeline

                    pipeline = st.session_state.mode1_pipeline

                    # Get prompt
                    from src.mode1_report_gen import PROMPT_VARIANTS
                    prompt = PROMPT_VARIANTS.get(prompt_choice)

                    t0 = time.time()
                    result = pipeline.run(
                        image=image,
                        prompt=prompt,
                        ground_truth_report=ground_truth or None,
                    )
                    elapsed = time.time() - t0

                    st.success(f"Report generated in {elapsed:.1f}s")

                    # Report output
                    st.markdown("#### 📋 Generated Report")
                    st.markdown(
                        f'<div class="report-box">{result["report"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # CLIP alignment
                    if show_clip_score and result.get("clip_alignment") is not None:
                        st.metric("CLIP Image-Text Alignment Score", f'{result["clip_alignment"]:.4f}',
                                  help="Cosine similarity between the X-ray image embedding and the generated report text embedding (CLIP). Higher = better alignment.")

                    # Metrics
                    if result.get("metrics"):
                        st.markdown("#### 📊 Evaluation Metrics")
                        m = result["metrics"]
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("BLEU-1", m.get("bleu1", "—"))
                        with c2: st.metric("BLEU-4", m.get("bleu4", "—"))
                        with c3: st.metric("ROUGE-L", m.get("rouge_l", "—"))
                        with c4: st.metric("BERTScore F1", m.get("bertscore_f1", "—"))

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

        elif generate_btn and not uploaded_file:
            st.warning("Please upload a chest X-ray image first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QA (RAG-Based)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload a Chest X-Ray + Ask a Clinical Question → Get a Grounded Answer")
    st.caption("ColPali retrieves similar cases from the MIMIC-CXR index. MedGemma answers using the retrieved context.")

    col_input, col_answer = st.columns([1, 1.4], gap="large")

    with col_input:
        qa_image_file = st.file_uploader(
            "Upload X-ray image",
            type=["png", "jpg", "jpeg"],
            key="mode2_upload",
        )

        question = st.text_input(
            "Clinical question",
            placeholder="e.g. Is there any evidence of pneumonia in this X-ray?",
        )

        # Example questions
        st.markdown("**Example questions:**")
        example_qs = [
            "Is there cardiomegaly?",
            "Is there a pleural effusion?",
            "Is there any pneumothorax?",
            "Where is the consolidation located?",
            "Are the lung fields clear?",
        ]
        for eq in example_qs:
            if st.button(eq, key=f"eq_{eq}", use_container_width=False):
                question = eq
                st.session_state["qa_question"] = eq

        show_context = st.checkbox("Show retrieved context", value=True)
        show_scores  = st.checkbox("Show retrieval scores", value=True)

        ask_btn = st.button("💬 Ask", type="primary", use_container_width=True)

    with col_answer:
        if qa_image_file:
            qa_image = Image.open(qa_image_file).convert("RGB")
            st.image(qa_image, caption="Query X-ray", use_container_width=True)

        if ask_btn:
            if not qa_image_file:
                st.warning("Please upload a chest X-ray image.")
            elif not question:
                st.warning("Please enter a clinical question.")
            else:
                qa_image = Image.open(qa_image_file).convert("RGB")

                with st.spinner("Encoding image → retrieving cases → generating answer…"):
                    try:
                        # Lazy load QA pipeline
                        if st.session_state.qa_pipeline is None:
                            from src.mode2_qa import QAPipeline
                            qa_pipeline = QAPipeline(
                                retrieval_backend=retrieval_backend,
                                top_k=top_k,
                                medgemma_load_in_4bit=load_in_4bit,
                            )
                            qa_pipeline.load_models()
                            # Attempt to load index from disk
                            try:
                                qa_pipeline.load_index()
                                st.session_state.index_loaded = True
                            except FileNotFoundError:
                                st.warning(
                                    "⚠️ No FAISS index found on disk. "
                                    "Run `notebooks/03_qa_pipeline.ipynb` to build the index first. "
                                    "Answering without retrieval context for now."
                                )
                                st.session_state.index_loaded = False

                            st.session_state.qa_pipeline = qa_pipeline

                        qa_pipeline = st.session_state.qa_pipeline

                        if st.session_state.index_loaded:
                            t0 = time.time()
                            result = qa_pipeline.run(image=qa_image, question=question)
                            elapsed = time.time() - t0

                            st.success(f"Answer generated in {elapsed:.1f}s")

                            # Answer
                            st.markdown("#### 💡 Answer")
                            st.markdown(
                                f'<div class="answer-box">{result["answer"]}</div>',
                                unsafe_allow_html=True,
                            )

                            # Retrieved context
                            if show_context:
                                st.markdown("#### 🔍 Retrieved Context")
                                st.markdown(
                                    f'<div class="context-box">{result["retrieved_context"]}</div>',
                                    unsafe_allow_html=True,
                                )

                            # Retrieval scores
                            if show_scores and result.get("retrieved_results"):
                                st.markdown("#### 📈 Retrieval Scores")
                                scores_data = [
                                    {"Rank": i+1, "Image": r.get("image_path", "")[-40:], "Score": f'{r["score"]:.4f}'}
                                    for i, r in enumerate(result["retrieved_results"])
                                ]
                                st.table(scores_data)

                        else:
                            # No index — answer directly from image without RAG
                            from src.models.medgemma import MedGemmaModel
                            if not hasattr(qa_pipeline, "_medgemma") or qa_pipeline._medgemma is None:
                                st.error("MedGemma not loaded.")
                            else:
                                answer = qa_pipeline._medgemma.answer_question(
                                    image=qa_image,
                                    question=question,
                                    context="",
                                )
                                st.markdown("#### 💡 Answer (no retrieval — direct inference)")
                                st.markdown(
                                    f'<div class="answer-box">{answer}</div>',
                                    unsafe_allow_html=True,
                                )

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.exception(e)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("DSAI 413 — Assignment 2 | Multi-Modal Chest X-Ray Intelligence System | Zewail City")
