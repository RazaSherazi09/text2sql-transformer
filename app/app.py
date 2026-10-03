import os
import sys
import time
import torch
import sentencepiece as spm
import gradio as gr
from huggingface_hub import hf_hub_download

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Handle optional local paths if running from Kaggle clone
if os.path.exists("/kaggle/working/text2sql_local"):
    sys.path.append("/kaggle/working/text2sql_local")

from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer
from decode import greedy_decode, beam_search_decode, parse_sql_string, format_readable_sql

REPO_ID = "razaasherazi/text2sql-transformer"
CKPT_PATH = "best_model.pt"
SP_PATH = "sql_sp.model"

# Auto-download model weights and tokenizer if missing
if not os.path.exists(SP_PATH):
    print("--> Downloading tokenizer from Hugging Face...")
    hf_hub_download(repo_id=REPO_ID, filename=SP_PATH, local_dir=".")

if not os.path.exists(CKPT_PATH):
    print("--> Downloading best_model.pt from Hugging Face...")
    hf_hub_download(repo_id=REPO_ID, filename=CKPT_PATH, local_dir=".")

# Load SentencePiece & Transformer Model
sp = spm.SentencePieceProcessor(model_file=SP_PATH)
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

checkpoint = torch.load(CKPT_PATH, map_location="cpu")
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

def generate_sql(question, columns, decoding_strategy, beam_size, max_length):
    if not question or not question.strip() or not columns or not columns.strip():
        return "-- Error: Please provide both a question and table column headers.", "", "{}", ""

    headers = [c.strip() for c in columns.split(",") if c.strip()]
    cols_formatted = " ".join(f"<c{i}> {name}" for i, name in enumerate(headers))
    src_text = f"{question.strip()} <sep> {cols_formatted}".lower()

    src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

    start_time = time.perf_counter()
    if decoding_strategy == "Greedy":
        pred_ids, _ = greedy_decode(model, src_ids, max_len=int(max_length))
    else:
        pred_ids, _ = beam_search_decode(model, src_ids, beam_size=int(beam_size), max_len=int(max_length))
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    cleaned_ids = [t for t in pred_ids if t not in (BOS_ID, EOS_ID)]
    pred_text = sp.decode(cleaned_ids)
    parsed = parse_sql_string(pred_text)
    readable_sql = format_readable_sql(parsed, headers)

    metadata = (
        f"Latency: {elapsed_ms:.1f} ms | Strategy: {decoding_strategy} "
        f"{f'(k={beam_size})' if decoding_strategy == 'Beam Search' else ''} | "
        f"Tokens: {len(cleaned_ids)}"
    )

    return readable_sql, pred_text, str(parsed), metadata

# High-contrast CSS fixing all input text, placeholders, and output code colors
HIGH_CONTRAST_CSS = """
/* Input text boxes */
input[type="text"], textarea, .gr-textbox input, .gr-textbox textarea {
    color: #0f172a !important;
    background-color: #ffffff !important;
    font-weight: 500 !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Placeholders */
input::placeholder, textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Focused inputs */
input[type="text"]:focus, textarea:focus {
    border-color: #4f46e5 !important;
    outline: 2px solid rgba(79, 70, 229, 0.2) !important;
}

/* Code block output styling */
.gr-code pre, .gr-code code, pre code, .code-wrap pre {
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    font-size: 0.95rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    padding: 14px !important;
    border-radius: 8px !important;
}

/* Labels & text headings */
label, .gr-form label, span.text-gray-500 {
    color: #1e293b !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=HIGH_CONTRAST_CSS, title="Text-to-SQL Transformer Studio") as demo:
    gr.Markdown(
        """
        # ⚡ Text-to-SQL Transformer Studio
        *Translate natural English questions over table schemas into executable SQL queries via a custom Transformer built from scratch.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Specification")
            input_question = gr.Textbox(
                label="Natural Language Question",
                placeholder="e.g. What is Terrence Ross' nationality?",
                value="What is Terrence Ross' nationality?"
            )
            input_columns = gr.Textbox(
                label="Table Column Headers (Comma-separated)",
                placeholder="e.g. Player, No., Nationality, Position",
                value="Player, No., Nationality, Position, Years in Toronto, School/Club Team"
            )

            with gr.Accordion("⚙️ Inference Settings", open=False):
                strategy = gr.Radio(
                    choices=["Greedy", "Beam Search"],
                    value="Greedy",
                    label="Decoding Strategy"
                )
                beam_slider = gr.Slider(
                    minimum=2, maximum=8, value=4, step=1,
                    label="Beam Size (k)"
                )
                max_len_slider = gr.Slider(
                    minimum=16, maximum=128, value=64, step=8,
                    label="Max Sequence Length"
                )

            btn = gr.Button("⚡ Generate SQL Query", variant="primary")

            gr.Examples(
                examples=[
                    [
                        "What is Terrence Ross' nationality?",
                        "Player, No., Nationality, Position, Years in Toronto, School/Club Team",
                        "Greedy", 4, 64
                    ],
                    [
                        "What is the highest attendance recorded?",
                        "Game, Date, Opponent, Result, Attendance",
                        "Greedy", 4, 64
                    ],
                    [
                        "How many cities have a population greater than 500000?",
                        "Rank, City, State, Population, Area",
                        "Beam Search", 4, 64
                    ]
                ],
                inputs=[input_question, input_columns, strategy, beam_slider, max_len_slider]
            )

        with gr.Column(scale=1):
            gr.Markdown("### 2. Synthesized SQL & Diagnostics")
            out_sql = gr.Code(label="Executable SQL", language="sql")
            out_meta = gr.Textbox(label="Execution Diagnostics", interactive=False)
            
            with gr.Accordion("🔍 Raw Model Tokens & AST", open=False):
                out_raw = gr.Textbox(label="Raw Vocabulary Output", interactive=False)
                out_ast = gr.Textbox(label="Parsed Schema AST Dictionary", interactive=False)

    btn.click(
        fn=generate_sql,
        inputs=[input_question, input_columns, strategy, beam_slider, max_len_slider],
        outputs=[out_sql, out_raw, out_ast, out_meta]
    )

    gr.Markdown(
        """
        ---
        <small style="color: #64748B;">
        Architecture: Seq2Seq Transformer (7.58M params, d_model=256, 4 heads, 3 layers) • WikiSQL Benchmark Baseline
        </small>
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)