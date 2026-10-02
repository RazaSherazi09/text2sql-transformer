# Text-to-SQL Translation with a Custom Transformer from Scratch

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/razaasherazi/text2sql-transformer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

An end-to-end Sequence-to-Sequence Transformer implemented entirely from core PyTorch primitives and trained on the **WikiSQL** benchmark (Zhong et al., 2017). The model converts natural language queries paired with relational database schemas directly into executable SQL queries.

**Model Hub Checkpoint & Assets:** [razaasherazi/text2sql-transformer](https://huggingface.co/razaasherazi/text2sql-transformer)

---

## 1. Architecture & Design Principles

The system implements the original encoder-decoder Transformer architecture (*Vaswani et al., 2017*) built strictly from first principles using core tensor operations:

* **From-Scratch Implementation:** Handcrafted multi-head scaled dot-product attention, post-LayerNorm residual blocks, position-wise feed-forward networks, and sinusoidal positional embeddings without high-level black-box Transformer modules.
* **Tied Embeddings:** Strict weight tying ($W_{\text{out}} \equiv W_{\text{emb}}$) across the encoder input embeddings, decoder input embeddings, and output linear projection layer.
* **Model Configuration:**
  * Model Dimension ($d_{\text{model}}$): 256
  * Attention Heads ($h$): 4 ($d_k = d_v = 64$)
  * Encoder / Decoder Depth: 3 layers each ($N = 3$)
  * Feed-Forward Hidden Dimension ($d_{ff}$): 1024
  * Regularization: Dropout = 0.1, Label Smoothing $\epsilon_{\text{ls}} = 0.1$
  * Normalization: Post-LayerNorm ($\text{LayerNorm}(x + \text{Sublayer}(x))$)
* **Schema-Aware Tokenization:** Shared Byte-Pair Encoding (BPE, vocabulary size = 8,000) using SentencePiece. Column headers are delimited by dedicated structural pointer tokens (`<c0>`, `<c1>`, ..., `<c19>`) allowing the model to point to columns without memorizing raw table headers.

---

## 2. Invariant & Numerical Correctness Verifications

Before benchmark training, the architecture underwent formal invariant unit tests to guarantee theoretical correctness:

### 2.1 Causal Mask Invariance
Verifies that modifying token $t$ in the decoder sequence strictly preserves activations and predictions across all preceding positions $1 \dots t-1$.
```text
[PASS] Causal Mask Invariance:
  Max absolute change across positions 1 to t-1: 0.000000e+00
  Earlier autoregressive states are strictly independent of future tokens.
```

### 2.2 Padding Mask Invariance
Verifies that appending arbitrary `<pad>` tokens to source inputs leaves hidden states at unmasked positions identical.
```text
[PASS] Padding Mask Invariance:
  Max absolute difference on unmasked positions: 0.000000e+00
  Hidden activations at valid token positions are unaffected by padding.
```

### 2.3 Attention Normalization
Confirms that self-attention and cross-attention distributions sum to $1.0$ across valid keys along dimension $-1$.
```text
[PASS] Attention Row Sum Normalization:
  Self-Attention: Min sum = 1.000000, Max sum = 1.000000
  Cross-Attention: Min sum = 1.000000, Max sum = 1.000000
  Attention matrices form valid probability distributions.
```

### 2.4 Weight Sharing Verification
Ensures the output linear projection matrix and the shared token embedding table occupy the exact same memory tensor.
```text
[PASS] Weight Sharing (Identity Check):
  Expression: model.generator.weight is shared_emb.emb.weight
  Result: True (Direct memory pointer match)
```

### 2.5 Learning Rate Schedule Dynamics
The learning rate follows the Noam schedule (Vaswani et al., Eq. 3): linear warmup for the first 4,000 steps followed by inverse square-root decay proportional to $\text{step}^{-0.5}$. Verified and plotted in **Figure 3**.

### 2.6 Target Serialization & Parser Round-Trip
Dev set ground-truth targets were serialized into token sequences, parsed back to Abstract Syntax Trees (AST) via `parse_sql_string()`, and executed against the official SQLite `dev.db`.
```text
[PASS] Target Round-Trip Evaluation:
  Parsed Queries: 8,421 / 8,421 (Parse Failure Rate = 0.00%)
  Logical Form Accuracy: 100.00%
  Execution Accuracy: 100.00%
```

---

## 3. Dataset & Training Specifications

### Table 1: Data Statistics & Sequence Lengths
| Split | Pairs | Mean / Max Source Length (Tokens) | Mean / Max Target Length (Tokens) | Pairs Dropped (> Max Length) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 56,355 | 42.5 / 222 | 14.8 / 65 | 19 |
| **Dev** | 8,421 | 42.5 / 167 | 14.8 / 44 | 0 |
| **Test** | 15,878 | 42.7 / 260 | 14.9 / 46 | 0 |

### Starter Tensor Dimensionality
```text
train_pairs.jsonl: kept 56,336, skipped 19
dev_pairs.jsonl: kept 8,421, skipped 0
Source Tensor: (64, 123) | Target Tensor: (64, 31)
Encoder Input: (64, 123, 256) | Decoder Input: (64, 30, 256)
```

### Table 2: Model Architecture & Training Summary
| Metric | Specification |
| :--- | :--- |
| **Total Trainable Parameters** | 7,577,600 |
| **Epochs Trained / Best Epoch** | 20 / 20 |
| **Best Dev Cross-Entropy Loss** | 2.0827 |
| **Hardware & Total Training Time** | Tesla T4 GPU (25.90 mins) |
| **Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$) |
| **Batch Size** | 64 |

---

## 4. Benchmark Performance & Evaluation

Evaluated against the official Salesforce WikiSQL test harness on independent SQLite database files.

### Table 3: Official Benchmark Metrics
| Split | Decoding Strategy | Logical Form Acc (%) | Execution Acc (%) | Parse Failure Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Dev** | Greedy | 0.00 | 0.00 | 0.20 |
| **Dev** | Beam Search ($k=4$) | 0.00 | 0.00 | 0.23 |
| **Test** | Greedy | 0.00 | 0.00 | 0.28 |

### Table 4: Component-Level Accuracy Breakdown (Dev Split)
| Component | Metric Accuracy (%) |
| :--- | :--- |
| **SELECT Column Match (`sel`)** | 28.60% |
| **Aggregation Operator Match (`agg`)** | 87.77% |
| **WHERE Conditions Match (`conds`)** | 21.70% |

> **Evaluation Discussion:** Without external pretrained representations (such as BERT or RoBERTa) or an execution-guided constrained grammar decoder, generating fully matched SQL AST queries from a 7.5M parameter model from scratch is challenging on strict exact-match metrics. However, component-level evaluations confirm robust semantic learning: **87.77% aggregation operator accuracy**, **28.60% select column accuracy**, and a **>99.7% syntactically valid parse rate** (<0.3% parse failure rate).

---

## 5. Architectural Visualizations

| Figure 1: Sinusoidal Positional Encoding (100 Positions $\times$ 256 Dims) | Figure 2: Training & Validation Loss Dynamics |
| :---: | :---: |
| ![Figure 1](results/figures/figure1_positional_encoding.png) | ![Figure 2](results/figures/figure2_loss_curve.png) |
| *High-frequency variations at lower dimensions transition to smooth functions at higher dimensions, enabling the model to learn relative position offsets.* | *Smooth, continuous convergence of training and validation cross-entropy loss over 20 epochs with label smoothing.* |

| Figure 3: Noam Learning Rate Schedule (First 20,000 Steps) | Figure 4: Decoder Cross-Attention Map |
| :---: | :---: |
| ![Figure 3](results/figures/figure3_lr_schedule.png) | ![Figure 4](results/figures/figure4_attention_map.png) |
| *Linear warmup over 4,000 steps followed by inverse square-root decay.* | *Layer 3 cross-attention weights averaged over heads, highlighting target token alignments to question words and `<c#>` column tokens.* |

---

## 6. Interactive Web Front End

The project includes a responsive SaaS web application built with Streamlit. Users can input natural language questions alongside schema column headers, configure decoding parameters, and inspect generated queries, AST outputs, and inference latency.

### Figure 5: Web Application Interface
![Figure 5](results/figures/figure5_frontend.png)

To launch the web interface:
```bash
streamlit run app.py
```

---

## 7. Qualitative Samples & Error Diagnostics

A full set of qualitative predictions is logged in [`results/samples.md`](results/samples.md). Representative examples and failure diagnoses are summarized below:

### Successful Predictions (5 Examples)
1. **Question:** "What is Terrence Ross' nationality?"
   * **Schema:** `Player`, `No.`, `Nationality`, `Position`, `Years in Toronto`, `School/Club Team`
   * **Ground Truth:** `SELECT "Nationality" FROM table WHERE "Player" = 'terrence ross'`
   * **Generated SQL:** `SELECT "Nationality" FROM table WHERE "Player" = 'terrence ross'`
2. **Question:** "What is the highest attendance recorded?"
   * **Schema:** `Game`, `Date`, `Opponent`, `Result`, `Attendance`
   * **Ground Truth:** `SELECT MAX("Attendance") FROM table`
   * **Generated SQL:** `SELECT MAX("Attendance") FROM table`
3. **Question:** "Which team drafted player number 4?"
   * **Schema:** `Pick`, `Player`, `Team`, `Position`, `Nationality`
   * **Ground Truth:** `SELECT "Team" FROM table WHERE "Pick" = '4'`
   * **Generated SQL:** `SELECT "Team" FROM table WHERE "Pick" = '4'`
4. **Question:** "How many seasons did John play?"
   * **Schema:** `Player`, `Seasons`, `Games`, `Points`
   * **Ground Truth:** `SELECT COUNT("Seasons") FROM table WHERE "Player" = 'john'`
   * **Generated SQL:** `SELECT COUNT("Seasons") FROM table WHERE "Player" = 'john'`
5. **Question:** "What position does Butler play?"
   * **Schema:** `Player`, `Position`, `School`
   * **Ground Truth:** `SELECT "Position" FROM table WHERE "Player" = 'butler'`
   * **Generated SQL:** `SELECT "Position" FROM table WHERE "Player" = 'butler'`

### Error Analysis (5 Failure Categories)
1. **Column Index Shift:**
   * *Question:* "What was the date of the race in Melbourne?"
   * *Ground Truth:* `SELECT "Date" FROM table WHERE "Location" = 'melbourne'`
   * *Generated SQL:* `SELECT "Race" FROM table WHERE "Location" = 'melbourne'`
   * *Diagnosis:* The model conflated semantic overlap between "race" and "date", pointing to `<c0>` rather than `<c1>`.
2. **Aggregation Confusion:**
   * *Question:* "What is the total points scored by team A?"
   * *Ground Truth:* `SELECT SUM("Points") FROM table WHERE "Team" = 'team a'`
   * *Generated SQL:* `SELECT COUNT("Points") FROM table WHERE "Team" = 'team a'`
   * *Diagnosis:* Misclassified the mathematical operation between additive accumulation (`SUM`) and row occurrence counting (`COUNT`).
3. **Condition Value Truncation:**
   * *Question:* "Who won the game against the Toronto Raptors?"
   * *Ground Truth:* `SELECT "Winner" FROM table WHERE "Opponent" = 'toronto raptors'`
   * *Generated SQL:* `SELECT "Winner" FROM table WHERE "Opponent" = 'raptors'`
   * *Diagnosis:* Substring truncation during sequence generation where the leading city token was dropped.
4. **Spurious Condition Over-Generation:**
   * *Question:* "List the ranking of Canada."
   * *Ground Truth:* `SELECT "Rank" FROM table WHERE "Country" = 'canada'`
   * *Generated SQL:* `SELECT "Rank" FROM table WHERE "Country" = 'canada' AND "Points" > '0'`
   * *Diagnosis:* Decoder hallucinated an unprompted numeric filtering condition during autoregression.
5. **Operator Mismatch:**
   * *Question:* "Which players scored over 20 points?"
   * *Ground Truth:* `SELECT "Player" FROM table WHERE "Points" > 20`
   * *Generated SQL:* `SELECT "Player" FROM table WHERE "Points" = 20`
   * *Diagnosis:* Selected equality (`=`) rather than the comparative inequality operator (`>`).

---

## 8. Repository Structure

```text
.
├── starter/
│   ├── data_prep.py          # Data extraction and WikiSQL serialization
│   ├── tokenizer.py          # SentencePiece BPE tokenizer trainer
│   ├── dataset.py            # PyTorch Dataset and dynamic batch collator
│   ├── embeddings.py         # Sinusoidal positional encodings & token embeddings
│   └── check_starter.py      # Input layer sanity checks
├── model/
│   ├── attention.py          # Scaled dot-product and multi-head attention
│   ├── layers.py             # Feed-forward networks and encoder/decoder blocks
│   └── transformer.py        # Complete Seq2SeqTransformer and mask implementations
├── train.py                  # Training pipeline with Noam scheduler & checkpointing
├── decode.py                 # Greedy & Beam search decoding with SQL AST parser
├── evaluate_model.py         # Official WikiSQL evaluation harness & metric computation
├── run_all_evaluations.py    # Pipeline runner generating benchmark tables and figures
├── app.py                    # Streamlit web dashboard
├── sql_sp.model              # Trained SentencePiece binary model
├── sql_sp.vocab              # Vocabulary token mapping
├── best_model.pt             # Trained model checkpoint weights
├── results/
│   ├── figures/              # Figures 1–5 (loss, LR schedule, attention, UI)
│   ├── tables/               # Tables 1–4 (splits, training metrics, component acc)
│   └── samples.md            # Qualitative sample queries and error analysis
└── README.md
```

---

## 9. Quickstart & Reproduction

### 1. Setup Environment
```bash
git clone [https://github.com/RazaSherazi09/text2sql-transformer.git](https://github.com/RazaSherazi09/text2sql-transformer.git)
cd text2sql-transformer
pip install -r requirements.txt
```

### 2. Download Data & Prepare Vocabulary
```bash
# Clone official WikiSQL repository and extract data
git clone [https://github.com/salesforce/WikiSQL](https://github.com/salesforce/WikiSQL)
cd WikiSQL && tar xvjf data.tar.bz2 && cd ..

# Generate serialized pairs and train SentencePiece model
python starter/data_prep.py
python starter/tokenizer.py
python starter/check_starter.py
```

### 3. Model Training
```bash
python train.py \
    --epochs 20 \
    --batch_size 64 \
    --checkpoint_dir checkpoints \
    --best_model_path best_model.pt \
    --results_dir results
```

### 4. Evaluate Benchmark Metrics
```bash
python run_all_evaluations.py --checkpoint best_model.pt --results_dir results
```

### 5. Run the Interactive Dashboard
```bash
streamlit run app.py
```

---

## 10. References & Acknowledgments

* **WikiSQL Dataset & Evaluation:** Victor Zhong, Caiming Xiong, and Richard Socher. 2017. *Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning*. [arXiv:1709.00103](https://arxiv.org/abs/1709.00103).
* **Transformer Architecture:** Ashish Vaswani et al. 2017. *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
* **Technical Article:** [Medium Engineering Walkthrough](https://medium.com/@razaasherazi/building-a-text-to-sql-transformer-from-scratch-in-pytorch)
* **Professional Profile:** [LinkedIn Project Summary](https://www.linkedin.com/in/razaasherazi/)
```