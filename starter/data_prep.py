# import json
# from pathlib import Path

# DATA_DIR = Path("WikiSQL/data")
# AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
# COND_OPS = ["=", ">", "<"]
# MAX_COLS = 64  # column tokens <c0> ... <c63>

# def load_split(split):
#     """Return (examples, tables) for 'train', 'dev' or 'test'."""
#     tables = {}
#     with open(DATA_DIR / f"{split}.tables.jsonl", encoding="utf-8") as f:
#         for line in f:
#             t = json.loads(line)
#             tables[t["id"]] = t
#     with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
#         examples = [json.loads(line) for line in f]
#     return examples, tables

# def encode_source(question, header):
#     """question <sep> <c0> col name <c1> col name ..."""
#     cols = " ".join(f"<c{i}> {name}" for i, name in enumerate(header))
#     return f"{question.strip()} <sep> {cols}".lower()

# def encode_target(sql):
#     """{'sel','agg','conds'} -> 'select count <c2> where <c0> = kim manners'"""
#     out = ["select"]
#     if sql["agg"]:
#         out.append(AGG_OPS[sql["agg"]].lower())
#     out.append(f"<c{sql['sel']}>")
#     for i, (col, op, val) in enumerate(sql["conds"]):
#         out += ["where" if i == 0 else "and", f"<c{col}>", COND_OPS[op], str(val)]
#     return " ".join(out).lower()

# def build_pairs(split):
#     examples, tables = load_split(split)
#     pairs = []
#     for ex in examples:
#         header = tables[ex["table_id"]]["header"]
#         pairs.append({
#             "table_id": ex["table_id"],
#             "src": encode_source(ex["question"], header),
#             "tgt": encode_target(ex["sql"]),
#         })
#     return pairs

# if __name__ == "__main__":
#     for split in ["train", "dev", "test"]:
#         pairs = build_pairs(split)
#         with open(f"{split}_pairs.jsonl", "w", encoding="utf-8") as f:
#             for p in pairs:
#                 f.write(json.dumps(p, ensure_ascii=False) + "\n")
#         print(f"{split}: {len(pairs)} pairs")
#     print(pairs[0]["src"])
#     print(pairs[0]["tgt"])

"""
starter/data_prep.py (V2)
WikiSQL Data Preparation with Schema-Typing Support.
"""

import json
import os
from pathlib import Path

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    "select", "where", "and",
    "count", "max", "min", "avg", "sum",
    "=", ">", "<"
] + [f"<c{i}>" for i in range(20)]

AGG_MAP = {
    0: "",
    1: "max",
    2: "min",
    3: "count",
    4: "sum",
    5: "avg"
}

OP_MAP = {
    0: "=",
    1: ">",
    2: "<"
}


def load_split(split_name="train", data_dir="WikiSQL/data"):
    """
    Loads examples and table schema dictionaries from WikiSQL data directory.
    """
    examples_path = os.path.join(data_dir, f"{split_name}.jsonl")
    tables_path = os.path.join(data_dir, f"{split_name}.tables.jsonl")

    tables = {}
    with open(tables_path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tables[t["id"]] = t

    examples = []
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    return examples, tables


def encode_source(question: str, header: list, types: list = None) -> str:
    """
    V2 Typed Linearization:
    <question> | columns: <c0> (type) col0 <c1> (type) col1 ...
    """
    if types is not None and len(types) == len(header):
        cols = [f"<c{i}> ({types[i].lower()}) {h}" for i, h in enumerate(header)]
    else:
        cols = [f"<c{i}> {h}" for i, h in enumerate(header)]
    return f"{question.strip()} | columns: {' '.join(cols)}"


def encode_target(sql_dict: dict) -> str:
    """
    Linearizes a WikiSQL query dictionary into sequential tokens:
    select [agg] <c{sel}> [where <c{col}> {op} {val} [and ...]]
    """
    sel_idx = sql_dict["sel"]
    agg_op = AGG_MAP.get(sql_dict["agg"], "")
    
    parts = ["select"]
    if agg_op:
        parts.append(agg_op)
    parts.append(f"<c{sel_idx}>")

    conds = sql_dict.get("conds", [])
    if conds:
        parts.append("where")
        cond_strs = []
        for col_idx, op_idx, val in conds:
            op_sym = OP_MAP.get(op_idx, "=")
            cond_strs.append(f"<c{col_idx}> {op_sym} {str(val).strip()}")
        parts.append(" and ".join(cond_strs))

    return " ".join(parts)


def prepare_data_pairs(split_name="train", data_dir="WikiSQL/data", out_dir="data", max_src_len=300, max_tgt_len=100):
    """
    Generates paired JSONL dataset for Seq2Seq consumption with length filtering.
    """
    os.makedirs(out_dir, exist_ok=True)
    examples, tables = load_split(split_name, data_dir)
    out_file = os.path.join(out_dir, f"{split_name}_pairs.jsonl")

    kept = 0
    dropped = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in examples:
            table = tables[ex["table_id"]]
            header = table["header"]
            types = table.get("types", None)

            src = encode_source(ex["question"], header, types)
            tgt = encode_target(ex["sql"])

            # Length filtering
            if len(src.split()) > max_src_len or len(tgt.split()) > max_tgt_len:
                dropped += 1
                continue

            record = {
                "question": ex["question"],
                "table_id": ex["table_id"],
                "source": src,
                "target": tgt,
                "sql": ex["sql"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[{split_name}] Kept: {kept} | Dropped as too long: {dropped} -> Saved to {out_file}")
    return out_file


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        prepare_data_pairs(split)