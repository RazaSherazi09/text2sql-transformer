import streamlit as st
import torch
import sentencepiece as spm
import os
import sys
import time
from huggingface_hub import hf_hub_download

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer
from decode import greedy_decode, beam_search_decode, parse_sql_string, format_readable_sql

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL Transformer Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Light Theme CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #F7F8FC;
    --card-bg: #FFFFFF;
    --text-primary: #1E293B;
    --text-secondary: #64748B;
    --accent-start: #4F46E5;
    --accent-end: #7C3AED;
    --border-light: #E2E8F0;
    --chip-bg: #EEF2FF;
    --chip-text: #4338CA;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid var(--border-light) !important;
}

#MainMenu, header, footer {
    visibility: hidden !important;
    display: none !important;
}

h1, h2, h3, h4, p, span, label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

.custom-card {
    background: var(--card-bg);
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04);
    margin-bottom: 20px;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 20px;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 8px;
}

.badge-ready {
    background-color: #DCFCE7;
    color: #15803D;
    border: 1px solid #BBF7D0;
}

.badge-missing {
    background-color: #FEF3C7;
    color: #B45309;
    border: 1px solid #FDE68A;
}

.badge-tech {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
}

.col-chip {
    display: inline-block;
    background: var(--chip-bg);
    color: var(--chip-text);
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
    border: 1px solid #E0E7FF;
}

div.stButton > button:first-child {
    background: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.45) !important;
}

div[data-testid="stHorizontalBlock"] div.stButton > button {
    background: #F8FAFC !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    width: auto !important;
}

div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    background: #F1F5F9 !important;
    border-color: #94A3B8 !important;
    transform: none !important;
}

input[type="text"] {
    border-radius: 10px !important;
    border: 1px solid var(--border-light) !important;
    padding: 10px 14px !important;
    font-size: 0.9rem !important;
    background-color: #FFFFFF !important;
}

input[type="text"]:focus {
    border-color: var(--accent-start) !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border-light);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    color: var(--accent-start) !important;
    border-bottom: 2px solid var(--accent-start) !important;
}

.metric-box {
    background-color: #F8FAFC;
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path="best_model.pt", sp_path="sql_sp.model"):
    repo_id = "razaasherazi/text2sql-transformer"

    # Automatically fetch tokenizer from Hugging Face Model Hub if missing
    if not os.path.exists(sp_path):
        with st.spinner("Downloading tokenizer from Hugging Face..."):
            try:
                hf_hub_download(repo_id=repo_id, filename="sql_sp.model", local_dir=".")
            except Exception as e:
                st.error(f"Error downloading tokenizer: {e}")

    # Automatically fetch model weights from Hugging Face Model Hub if missing
    if not os.path.exists(checkpoint_path):
        with st.spinner("Downloading best_model.pt from Hugging Face..."):
            try:
                hf_hub_download(repo_id=repo_id, filename="best_model.pt", local_dir=".")
            except Exception as e:
                st.error(f"Error downloading checkpoint: {e}")

    if not os.path.exists(sp_path) or not os.path.exists(checkpoint_path):
        return None, None

    sp = spm.SentencePieceProcessor(model_file=sp_path)
    vocab_size = sp.get_piece_size()
    d_model = 256

    shared_emb = TokenEmbedding(vocab_size, d_model, pad_id=PAD_ID)
    enc_in = InputLayer(shared_emb, d_model=d_model, dropout=0.0)
    dec_in = InputLayer(shared_emb, d_model=d_model, dropout=0.0)

    model = Seq2SeqTransformer(
        enc_input_layer=enc_in,
        dec_input_layer=dec_in,
        shared_embedding=shared_emb,
        d_model=d_model,
        h=4,
        d_ff=1024,
        num_layers=3,
        dropout=0.0,
        pad_id=PAD_ID
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, sp


# Render Top Hero Section
def render_hero():
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px;">
                Text-to-SQL <span style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Transformer</span>
            </h1>
            <p style="font-size: 0.95rem; color: #64748B; margin-bottom: 14px;">
                Translating natural language queries into executable relational SQL queries via custom sequence-to-sequence attention.
            </p>
            <div>
                <span class="badge-pill badge-tech">Transformer from scratch</span>
                <span class="badge-pill badge-tech">SentencePiece Tokenizer</span>
                <span class="badge-pill badge-tech">Greedy & Beam Search</span>
                <span class="badge-pill badge-tech">WikiSQL Schema Format</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Render Model Status Badge
def render_status_badge(model_loaded: bool):
    if model_loaded:
        st.sidebar.markdown(
            '<div class="badge-pill badge-ready">● Model ready (best_model.pt)</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div class="badge-pill badge-missing">▲ Checkpoint missing</div>',
            unsafe_allow_html=True
        )


# Initialize Session State
if "question_text" not in st.session_state:
    st.session_state["question_text"] = "What is Terrence Ross' nationality?"
if "columns_text" not in st.session_state:
    st.session_state["columns_text"] = "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
if "results_data" not in st.session_state:
    st.session_state["results_data"] = None

# Model Loading
model, sp = load_model()

# Header Layout
render_hero()

# Sidebar: Controls & Info
with st.sidebar:
    st.markdown("### Model Runtime")
    render_status_badge(model is not None)

    st.markdown("---")
    st.markdown("### Inference Settings")
    decode_mode = st.radio("Decoding Strategy", ["Greedy", "Beam Search"], index=0)

    beam_size = 4
    if decode_mode == "Beam Search":
        beam_size = st.slider("Beam Size (k)", min_value=2, max_value=8, value=4)

    max_len = st.slider("Max Output Length", min_value=16, max_value=128, value=64, step=8)

    st.markdown("---")
    st.markdown("### Architecture Pipeline")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748B; line-height: 1.5;">
        1. <b>Tokenize:</b> SentencePiece maps question & schema to subwords.<br>
        2. <b>Encode:</b> Sinusoidal embeddings feed 3-layer Transformer encoder.<br>
        3. <b>Decode:</b> Autoregressive cross-attention yields SQL tokens.<br>
        4. <b>Parse:</b> SQL tokens reconstruct into executable SQLite syntax.
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Two-Column Layout
col_left, col_right = st.columns([1, 1], gap="large")

# Preset Samples
PRESETS = [
    {
        "label": "🏀 NBA Player",
        "q": "What is Terrence Ross' nationality?",
        "cols": "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
    },
    {
        "label": "🏆 CFL Attendance",
        "q": "What is the highest attendance recorded?",
        "cols": "Game, Date, Opponent, Result, Attendance"
    },
    {
        "label": "👥 Population",
        "q": "How many cities have a population greater than 500000?",
        "cols": "Rank, City, State, Population, Area"
    }
]

with col_left:
    st.markdown(
        """
        <div class="card-title">Query Specification</div>
        <div class="card-subtitle">Formulate natural language requests and specify table headers.</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748B;'>EXAMPLE PRESETS</span>", unsafe_allow_html=True)
    preset_cols = st.columns(len(PRESETS))
    for idx, preset in enumerate(PRESETS):
        if preset_cols[idx].button(preset["label"], key=f"preset_{idx}"):
            st.session_state["question_text"] = preset["q"]
            st.session_state["columns_text"] = preset["cols"]
            st.rerun()

    question = st.text_input(
        "Natural Language Question",
        value=st.session_state["question_text"],
        key="input_q"
    )

    columns_input = st.text_input(
        "Table Headers (Comma-separated)",
        value=st.session_state["columns_text"],
        key="input_cols"
    )

    parsed_header_chips = [c.strip() for c in columns_input.split(",") if c.strip()]
    if parsed_header_chips:
        chips_html = "".join([f'<span class="col-chip">{c}</span>' for c in parsed_header_chips])
        st.markdown(f"<div style='margin-bottom: 18px;'>{chips_html}</div>", unsafe_allow_html=True)

    generate_clicked = st.button("Generate SQL Query ⚡")

    if generate_clicked:
        if not question.strip() or not columns_input.strip():
            st.error("Please supply both a natural question and table headers.")
        elif model is None:
            st.error("Model checkpoint not available. Please verify 'best_model.pt'.")
        else:
            headers = [c.strip() for c in columns_input.split(",") if c.strip()]
            cols_formatted = " ".join(f"<c{i}> {name}" for i, name in enumerate(headers))
            src_text = f"{question.strip()} <sep> {cols_formatted}".lower()
            src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

            start_t = time.perf_counter()
            with st.spinner("Decoding attention graph..."):
                if decode_mode == "Greedy":
                    pred_ids, _ = greedy_decode(model, src_ids, max_len=max_len)
                    mode_used = "Greedy"
                else:
                    pred_ids, _ = beam_search_decode(model, src_ids, beam_size=beam_size, max_len=max_len)
                    mode_used = f"Beam Search (k={beam_size})"

                cleaned_ids = [t for t in pred_ids if t not in (BOS_ID, EOS_ID)]
                pred_text = sp.decode(cleaned_ids)
                parsed = parse_sql_string(pred_text)
                sql_readable = format_readable_sql(parsed, headers)
            duration = time.perf_counter() - start_t

            st.session_state["results_data"] = {
                "pred_text": pred_text,
                "sql_readable": sql_readable,
                "parsed": parsed,
                "headers": headers,
                "mode": mode_used,
                "duration": duration,
                "tokens": len(cleaned_ids)
            }

with col_right:
    st.markdown(
        """
        <div class="card-title">Generated SQL Result</div>
        <div class="card-subtitle">Synthesized structured query and inspection artifacts.</div>
        """,
        unsafe_allow_html=True
    )

    res = st.session_state["results_data"]

    if res is None:
        st.markdown(
            """
            <div style="border: 2px dashed #E2E8F0; border-radius: 14px; padding: 48px 24px; text-align: center; color: #94A3B8;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div style="font-weight: 600; font-size: 0.95rem; color: #64748B;">No query generated yet</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">Choose an example preset or enter inputs on the left, then click Generate SQL.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        tab_sql, tab_raw, tab_details = st.tabs(["Formatted SQL", "Raw Tokens", "Execution Metadata"])

        with tab_sql:
            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748B;'>EXECUTABLE SQL</span>", unsafe_allow_html=True)
            st.code(res["sql_readable"], language="sql")

        with tab_raw:
            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748B;'>DECODED VOCABULARY TOKENS</span>", unsafe_allow_html=True)
            st.code(res["pred_text"])

        with tab_details:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Decoding Strategy</div>
                        <div class="metric-value">{res["mode"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Latency</div>
                        <div class="metric-value">{res["duration"]*1000:.1f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748B;'>PARSED SCHEMA AST</span>", unsafe_allow_html=True)
            st.json(res["parsed"])

# Footer: Architectural Specifications
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #94A3B8; padding: 4px 0;">
        <span>Seq2Seq Transformer &bull; d_model=256 &bull; 4 Attention Heads &bull; 3 Layers &bull; d_ff=1024</span>
        <span>WikiSQL Benchmark Baseline</span>
    </div>
    """,
    unsafe_allow_html=True
)