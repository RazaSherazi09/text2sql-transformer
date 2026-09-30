import torch
import os
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
    print(f"Running full evaluation pipeline on {device}...")

    # 1. Gold Round-Trip Check
    gold_roundtrip_check("dev")

    # 2. Load trained model
    model, sp = load_model("best_model.pt", "sql_sp.model", device=device)

    # 3. Generate Dev Greedy, Dev Beam (4), and Test Predictions
    dev_greedy_lf, dev_greedy_ex, dev_greedy_fail = generate_predictions_and_evaluate(
        model, sp, split="dev", method="greedy", output_path="results/dev_greedy.jsonl", device=device
    )
    dev_beam_lf, dev_beam_ex, dev_beam_fail = generate_predictions_and_evaluate(
        model, sp, split="dev", method="beam", output_path="results/dev_beam.jsonl", device=device
    )
    test_lf, test_ex, test_fail = generate_predictions_and_evaluate(
        model, sp, split="test", method="greedy", output_path="results/test.jsonl", device=device
    )

    # 4. Write Table 3: Official Metrics
    table3 = f"""# Table 3 – Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
| :--- | :--- | :--- | :--- | :--- |
| Dev | greedy | {dev_greedy_lf:.2f} | {dev_greedy_ex:.2f} | {dev_greedy_fail:.2f} |
| Dev | beam (4) | {dev_beam_lf:.2f} | {dev_beam_ex:.2f} | {dev_beam_fail:.2f} |
| Test | greedy | {test_lf:.2f} | {test_ex:.2f} | {test_fail:.2f} |
"""
    with open("results/tables/table3_official_metrics.md", "w") as f:
        f.write(table3)

    # 5. Component Accuracy (Table 4)
    # Compare Dev Greedy against gold
    examples, _ = load_split("dev")
    with open("results/dev_greedy.jsonl", "r", encoding="utf-8") as f:
        preds = [eval(line) for line in f]

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

    # 6. Qualitative Samples (results/samples.md)
    generate_samples_md(model, sp, device=device)

    # 7. Figure 4: Cross-attention Map
    plot_attention_map(model, sp, example_idx=0, device=device)

    print("\nAll deliverables generated in results/ directory!")

if __name__ == "__main__":
    main()