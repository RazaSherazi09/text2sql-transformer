import json
import os
import torch
from evaluate_model import (
    load_model,
    gold_roundtrip_check,
    generate_predictions_and_evaluate,
    generate_samples_md,
    plot_attention_map
)
from starter.data_prep import load_split

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing complete evaluation pipeline on: {device}")

    # 1. Gold Round-Trip Check (Section 3.2)
    gold_roundtrip_check("dev")

    # 2. Load best checkpoint
    model, sp = load_model("best_model.pt", "sql_sp.model", device=device)

    # 3. Generate Official Predictions (Task 4.3 / Table 3)
    dev_greedy_lf, dev_greedy_ex, dev_greedy_fail = generate_predictions_and_evaluate(
        model, sp, split="dev", method="greedy", output_path="results/dev_greedy.jsonl", device=device
    )
    dev_beam_lf, dev_beam_ex, dev_beam_fail = generate_predictions_and_evaluate(
        model, sp, split="dev", method="beam", output_path="results/dev_beam.jsonl", device=device
    )
    test_lf, test_ex, test_fail = generate_predictions_and_evaluate(
        model, sp, split="test", method="greedy", output_path="results/test.jsonl", device=device
    )

    # 4. Generate Table 3 Markdown
    table3 = f"""# Table 3 – Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
| :--- | :--- | :--- | :--- | :--- |
| Dev | greedy | {dev_greedy_lf:.2f} | {dev_greedy_ex:.2f} | {dev_greedy_fail:.2f} |
| Dev | beam (4) | {dev_beam_lf:.2f} | {dev_beam_ex:.2f} | {dev_beam_fail:.2f} |
| Test | greedy | {test_lf:.2f} | {test_ex:.2f} | {test_fail:.2f} |
"""
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/table3_official_metrics.md", "w") as f:
        f.write(table3)
    print("Saved: results/tables/table3_official_metrics.md")

    # 5. Component Accuracy on Dev (Table 4)
    examples, _ = load_split("dev")
    with open("results/dev_greedy.jsonl", "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]

    total = len(examples)
    sel_ok, agg_ok, where_ok = 0, 0, 0
    for ex, p in zip(examples, preds):
        if "error" in p:
            continue
        g = ex["sql"]
        q = p["query"]
        if g["sel"] == q["sel"]:
            sel_ok += 1
        if g["agg"] == q["agg"]:
            agg_ok += 1
        g_c = sorted([[c[0], c[1], str(c[2]).strip().lower()] for c in g["conds"]])
        p_c = sorted([[c[0], c[1], str(c[2]).strip().lower()] for c in q["conds"]])
        if g_c == p_c:
            where_ok += 1

    table4 = f"""# Table 4 – Component accuracy (dev)

| Component | Accuracy (%) |
| :--- | :--- |
| sel column correct | {(sel_ok / total) * 100:.2f} |
| agg correct | {(agg_ok / total) * 100:.2f} |
| WHERE clause correct | {(where_ok / total) * 100:.2f} |
"""
    with open("results/tables/table4_component_accuracy.md", "w") as f:
        f.write(table4)
    print("Saved: results/tables/table4_component_accuracy.md")

    # 6. Qualitative Samples (Task 4.6)
    generate_samples_md(model, sp, device=device)

    # 7. Cross-Attention Map (Figure 4)
    plot_attention_map(model, sp, example_idx=0, device=device)

    print("\nALL TASKS COMPLETE: All tables, figures, and samples are populated in results/.")


if __name__ == "__main__":
    main()