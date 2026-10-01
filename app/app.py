import streamlit as st
import torch
import sentencepiece as spm
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer
from decode import greedy_decode, beam_search_decode, parse_sql_string, format_readable_sql

st.set_page_config(page_title="Text-to-SQL Transformer", layout="centered")

@st.cache_resource
def load_model(checkpoint_path="best_model.pt", sp_path="sql_sp.model"):
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sp

st.title("Natural Language to SQL Transformer")
st.markdown(
    "Custom Transformer built from scratch without pretrained weights to convert "
    "natural language questions over database tables into executable SQL queries."
)

model, sp = load_model()

if model is None:
    st.warning("Model checkpoint (`best_model.pt`) or tokenizer (`sql_sp.model`) not found. Run training on Kaggle/Colab first and place the artifacts in the root directory.")
else:
    st.success("Trained model and SentencePiece tokenizer loaded successfully.")

question = st.text_input(
    "Enter natural language question:",
    value="What is Terrence Ross' nationality?"
)

columns_input = st.text_input(
    "Enter comma-separated table column names:",
    value="Player, No., Nationality, Position, Years in Toronto, School/Club Team"
)

decode_mode = st.radio("Decoding Strategy:", ["Greedy", "Beam Search (k=4)"], horizontal=True)

if st.button("Generate SQL"):
    if not question.strip() or not columns_input.strip():
        st.error("Please provide both a question and table column headers.")
    elif model is None:
        st.error("Model is not loaded. Cannot perform inference.")
    else:
        # Preprocess input into the format expected by the model
        headers = [c.strip() for c in columns_input.split(",") if c.strip()]
        cols_formatted = " ".join(f"<c{i}> {name}" for i, name in enumerate(headers))
        src_text = f"{question.strip()} <sep> {cols_formatted}".lower()

        src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

        with st.spinner("Generating SQL query..."):
            if decode_mode == "Greedy":
                pred_ids, _ = greedy_decode(model, src_ids, max_len=64)
            else:
                pred_ids, _ = beam_search_decode(model, src_ids, beam_size=4, max_len=64)

            cleaned_ids = [t for t in pred_ids if t not in (BOS_ID, EOS_ID)]
            pred_text = sp.decode(cleaned_ids)
            parsed = parse_sql_string(pred_text)
            sql_readable = format_readable_sql(parsed, headers)

        st.subheader("Results")
        st.markdown("**Raw Model Output:**")
        st.code(pred_text)

        st.markdown("**Executable SQL (with real column names):**")
        st.code(sql_readable, language="sql")

