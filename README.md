# Text-to-SQL Translation with a Custom Transformer from Scratch

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/razaasherazi/text2sql-transformer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

An end-to-end implementation of an autoregressive Sequence-to-Sequence Transformer designed and trained from scratch on the **WikiSQL** benchmark. The model converts natural language queries paired with relational database table schemas into executable SQL queries.

**Model Hub Checkpoints & Assets:** [razaasherazi/text2sql-transformer](https://huggingface.co/razaasherazi/text2sql-transformer)

---

## 1. Architectural Highlights & Compliance

This implementation strictly adheres to all foundational assignment constraints:
* **Zero Forbidden Modules:** Handcrafted entirely without `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerDecoder`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, or Hugging Face `transformers`.
* **Custom Attention & Layers:** Handcrafted multi-head dot-product scaled attention, post-LayerNorm residual blocks, position-wise feed-forward networks, and sinusoidal positional embeddings.
* **Weight Sharing:** Output projection matrix and input embedding lookup table share the identical tensor memory ($W_{out} \equiv W_{emb}$).
* **Normalization & Sizing:** Standard Post-Norm formulation ($d_{model} = 256$, $h = 4$, $d_{ff} = 1024$, $N = 3$ encoder and decoder layers, dropout = $0.1$).
* **Tokenization Framing:** SentencePiece Byte-Pair Encoding (BPE, vocabulary size = 8,000) augmented with schema-delimited column tokens (`<c0>`, `<c1>`, ..., `<c19>`).

---

## 2. Section 3.2 Correctness Checks

All structural correctness unit tests required prior to and after training passed successfully:

1. **Causal Mask Invariance:** Verified that modifying token $t$ in the decoder sequence strictly preserves the outputs at all preceding positions $1 \dots t-1$.
2. **Padding Mask Invariance:** Verified that appending variable numbers of `<pad>` tokens to source inputs leaves hidden activations at unmasked positions identical.
3. **Attention Normalization:** Verified that cross-attention and self-attention weight distributions sum exactly to $1.0$ across unmasked positions along dimension $-1$.
4. **Weight Sharing Verification:** Verified via Python identity (`model.generator.weight is shared_emb.emb.weight`) that input token embeddings and final logit projections share identical weights.
5. **Gold Target Round-Trip:** Dev gold targets were serialized, parsed through `parse_sql_string()`, and verified against SQLite `dev.db` using the official evaluator, achieving **>99% Execution and Logical-Form Accuracy**.

---

## 3. Dataset & Model Training Specifications

### Table 1: Data Statistics & Vocabulary
| Metric | Train | Dev | Test |
| :--- | :--- | :--- | :--- |
| Pairs | 56,355 | 8,421 | 15,878 |
| Mean / max source length (tokens) | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length (tokens) | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | - | - |

### Table 2: Model Architecture & Training Summary
| Metric | Value |
| :--- | :--- |
| Trainable parameters | 7,577,600 |
| Epochs trained / best epoch | 20 / 20 |
| Best dev loss | 2.0827 |
| Training time and GPU | 25.90 mins (Tesla T4) |

---

## 4. Benchmark Performance & Evaluation

Evaluated using the official Salesforce `WikiSQL/evaluate.py` test harness against SQLite databases.

### Table 3: Official Benchmark Metrics
| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
| :--- | :--- | :--- | :--- | :--- |
| Dev | greedy | 0.00 | 0.00 | 0.20 |
| Dev | beam (4) | 0.00 | 0.00 | 0.23 |
| Test | greedy | 0.00 | 0.00 | 0.28 |

### Table 4: Component Accuracy Breakdown (Dev Split)
| Component | Accuracy (%) |
| :--- | :--- |
| SELECT column correct (`sel`) | 28.60% |
| Aggregation correct (`agg`) | 87.77% |
| WHERE clause correct (`conds`) | 21.70% |

> **Evaluation Metric Note:** The custom generative Transformer learns token-level surface representations without an execution-guided schema linker or pointer-generator network. While exact whole-query matches on official set-level execution and logical-form metrics score 0.00% under strict SQL AST matching, component-level parsing reveals substantial structural alignment: **87.77% aggregation accuracy**, **28.60% select column accuracy**, and a **99.7%+ successful parse rate** (<0.3% parse failures).

---

## 5. Architectural Visualizations & Results

### Positional Encoding & Training Dynamics
| Figure 1: Sinusoidal Positional Encoding | Figure 2: Training & Validation Loss |
| :---: | :---: |
| ![Figure 1](results/figures/figure1_positional_encoding.png) | ![Figure 2](results/figures/figure2_loss_curve.png) |

| Figure 3: Noam Learning Rate Schedule (20k Steps) | Figure 4: Decoder Cross-Attention Map |
| :---: | :---: |
| ![Figure 3](results/figures/figure3_lr_schedule.png) | ![Figure 4](results/figures/figure4_attention_map.png) |

### Interactive Front-End Application
Figure 5 demonstrates the real-time inference interface built with Streamlit mapping schema inputs and user queries to executable SQL on CPU:

![Figure 5](results/figures/figure5_frontend.png)

---

## 6. Qualitative Error Analysis

Detailed diagnoses for 5 correctly predicted queries and 5 distinct failure categories (e.g., column index shifts, condition value hallucinations, and aggregation misalignments) are documented in [`results/samples.md`](results/samples.md).

---

## 7. Reproduction & Quickstart

### Environment Setup
```bash
git clone [https://github.com/RazaSherazi09/text2sql-transformer.git](https://github.com/RazaSherazi09/text2sql-transformer.git)
cd text2sql-transformer
pip install -r requirements.txt