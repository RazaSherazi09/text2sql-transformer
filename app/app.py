import streamlit as st
import torch
import sentencepiece as spm
import os
import sys
import time
import html
from huggingface_hub import hf_hub_download

# Ensure current directory and root are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

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

# High-contrast CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

#MainMenu, header, footer {
    visibility: hidden !important;
    display: none !important;
}

h1, h2, h3, h4, p, span, label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #0F172A !important;
}

/* High-contrast Inputs */
input[type="text"], .stTextInput input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

input[type="text"]:focus, .stTextInput input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}

input::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

/* High-contrast Action Button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.45) !important;
}

/* Chips for Column Names */
.col-chip {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px 6px 4px 0;
    border: 1px solid #E0E7FF;
}

.metric-box {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 4px;
}

/* Custom Crystal Clear Terminal / Code Box */
.terminal-card {
    background: #0F172A;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    border: 1px solid #1E293B;
    margin-top: 10px;
}

.terminal-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #334155;
}

.dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.dot-red { background: #EF4444; }
.dot-yellow { background: #F59E0B; }
.dot-green { background: #10B981; }

.sql-content {
    color: #38BDF8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    line-height: 1.6;
    word-break: break-word;
    white-space: pre-wrap;
}

.sql-keyword {
    color: #F43F5E;
    font-weight: 700;
}
.sql-column {
    color: #FBBF24;
}
.sql-string {
    color: #34D399;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path="best_model.pt", sp_path="sql_sp.model"):
    repo_id = "razaasherazi/text2sql-transformer"

    if not os.path.exists(sp_path):
        with st.spinner("Downloading tokenizer from Hugging Face..."):
            try:
                hf_hub_download(repo_id=repo_id, filename="sql_sp.model", local_dir=".")
            except Exception as e:
                st.error(f"Error downloading tokenizer: {e}")

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


def highlight_sql(sql_str):
    # Safe HTML escape first
    escaped = html.escape(sql_str)
    keywords = ["SELECT", "FROM", "WHERE", "AND", "OR", "COUNT", "MAX", "MIN", "SUM", "AVG"]
    for kw in keywords:
        escaped = escaped.replace(kw, f'<span class="sql-keyword">{kw}</span>')
    return escaped


# Top Header
st.markdown(
    """
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px;">
            Text-to-SQL <span style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Transformer Studio</span>
        </h1>
        <p style="font-size: 0.95rem; color: #64748B; margin-bottom: 14px;">
            Translating natural language questions over table schemas into executable SQL queries via a custom Transformer built from scratch.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "question_text" not in st.session_state:
    st.session_state["question_text"] = "What is Terrence Ross' nationality?"
if "columns_text" not in st.session_state:
    st.session_state["columns_text"] = "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
if "results_data" not in st.session_state:
    st.session_state["results_data"] = None

model, sp = load_model()

# Sidebar: Controls
with st.sidebar:
    st.markdown("### Runtime Status")
    if model is not None:
        st.success("● Model Ready (`best_model.pt`)")
    else:
        st.warning("▲ Model Not Found")

    st.markdown("---")
    st.markdown("### Inference Settings")
    decode_mode = st.radio("Decoding Strategy", ["Greedy", "Beam Search"], index=0)

    beam_size = 4
    if decode_mode == "Beam Search":
        beam_size = st.slider("Beam Size (k)", min_value=2, max_value=8, value=4)

    max_len = st.slider("Max Output Length", min_value=16, max_value=128, value=64, step=8)

# Layout
col_left, col_right = st.columns([1, 1], gap="large")

PRESETS = [
    {
        "label": "🏀 NBA Player",
        "q": "What is Terrence Ross' nationality?",
        "cols": "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
    },
    {
        "label": "🏆 Attendance",
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
    st.markdown("### 1. Query Specification")
    st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #64748B;'>EXAMPLE PRESETS</span>", unsafe_allow_html=True)
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
            with st.spinner("Generating query..."):
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
    st.markdown("### 2. Synthesized SQL Result")
    res = st.session_state["results_data"]

    if res is None:
        st.info("Fill out the inputs on the left and click **Generate SQL Query ⚡**.")
    else:
        tab_sql, tab_raw, tab_details = st.tabs(["Formatted SQL", "Raw Tokens", "Metadata"])

        with tab_sql:
            st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #64748B;'>EXECUTABLE SQL</span>", unsafe_allow_html=True)
            formatted_highlighted = highlight_sql(res["sql_readable"])
            st.markdown(
                f"""
                <div class="terminal-card">
                    <div class="terminal-header">
                        <span class="dot dot-red"></span>
                        <span class="dot dot-yellow"></span>
                        <span class="dot dot-green"></span>
                        <span style="font-size: 0.72rem; color: #94A3B8; margin-left: 8px; font-family: monospace;">sqlite3 &bull; output.sql</span>
                    </div>
                    <div class="sql-content">{formatted_highlighted}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab_raw:
            st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #64748B;'>DECODED MODEL TOKENS</span>", unsafe_allow_html=True)
            escaped_raw = html.escape(res["pred_text"])
            st.markdown(
                f"""
                <div class="terminal-card">
                    <div class="terminal-header">
                        <span class="dot dot-red"></span>
                        <span class="dot dot-yellow"></span>
                        <span class="dot dot-green"></span>
                        <span style="font-size: 0.72rem; color: #94A3B8; margin-left: 8px; font-family: monospace;">sentencepiece &bull; raw_tokens</span>
                    </div>
                    <div class="sql-content" style="color: #A5B4FC !important;">{escaped_raw}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab_details:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Strategy</div>
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

            st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #64748B;'>PARSED SCHEMA AST</span>", unsafe_allow_html=True)
            st.json(res["parsed"])

st.markdown("---")
st.caption("Seq2Seq Transformer (7.58M parameters) • Built from scratch without pretrained weights • WikiSQL Benchmark")