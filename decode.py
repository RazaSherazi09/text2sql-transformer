import re
import json
import torch
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from starter.data_prep import AGG_OPS, COND_OPS
from model.transformer import make_padding_mask, make_causal_mask


def greedy_decode(model, src, max_len=64, device="cpu"):
    model.eval()
    src = src.to(device)
    src_mask = make_padding_mask(src, PAD_ID)
    
    with torch.no_grad():
        memory = model.encode(src, src_mask=src_mask)
        ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
        last_cross_attn = None
        
        for _ in range(max_len):
            tgt_mask = make_causal_mask(ys.size(1), device=device)
            out, cross_attn = model.decode(ys, memory, tgt_mask=tgt_mask, cross_mask=src_mask)
            logits = model.generator(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            last_cross_attn = cross_attn
            
            if next_token.item() == EOS_ID:
                break
                
    return ys.squeeze(0).tolist(), last_cross_attn


def beam_search_decode(model, src, beam_size=4, max_len=64, device="cpu"):
    model.eval()
    src = src.to(device)
    src_mask = make_padding_mask(src, PAD_ID)

    with torch.no_grad():
        memory = model.encode(src, src_mask=src_mask) # (1, S, D)
        
        beams = [([BOS_ID], 0.0)]
        completed = []

        for _ in range(max_len):
            active_beams = [b for b in beams if b[0][-1] != EOS_ID]
            finished_beams = [b for b in beams if b[0][-1] == EOS_ID]
            completed.extend(finished_beams)

            if not active_beams or len(completed) >= beam_size:
                break

            # Batch decode all active hypotheses at once
            B_act = len(active_beams)
            ys_batch = torch.tensor([b[0] for b in active_beams], dtype=torch.long, device=device)
            tgt_mask = make_causal_mask(ys_batch.size(1), device=device)
            mem_batch = memory.expand(B_act, -1, -1)
            src_mask_batch = src_mask.expand(B_act, -1, -1, -1)

            out, _ = model.decode(ys_batch, mem_batch, tgt_mask=tgt_mask, cross_mask=src_mask_batch)
            log_probs = torch.log_softmax(model.generator(out[:, -1, :]), dim=-1) # (B_act, V)

            candidates = []
            for i, (tokens, score) in enumerate(active_beams):
                topk_p, topk_idx = torch.topk(log_probs[i], beam_size)
                for p, idx in zip(topk_p.tolist(), topk_idx.tolist()):
                    candidates.append((tokens + [idx], score + p))

            candidates.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
            beams = candidates[:beam_size]

        if completed:
            completed.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
            best_tokens = completed[0][0]
        else:
            beams.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
            best_tokens = beams[0][0]

    return best_tokens, None


def restore_casing(val_lower, original_text):
    """
    Finds case-insensitive match of val_lower in original_text
    to restore proper nouns for SQLite queries.
    """
    if not original_text or not val_lower:
        return val_lower
    match = re.search(re.escape(val_lower), original_text, flags=re.IGNORECASE)
    return match.group(0) if match else val_lower


def parse_sql_string(sql_str, original_question=None):
    """
    Parses predicted string into WikiSQL JSON:
    {"sel": int, "agg": int, "conds": [[col, op, val], ...]}
    """
    try:
        raw_text = sql_str.strip()
        text = raw_text.lower()
        if not text.startswith("select "):
            return None

        if " where " in text:
            select_part, where_part = text.split(" where ", 1)
        else:
            select_part, where_part = text, ""

        sel_tokens = select_part.replace("select", "").strip().split()
        if not sel_tokens:
            return None

        agg_idx = 0
        agg_names = [op.lower() for op in AGG_OPS[1:]]
        if sel_tokens[0] in agg_names:
            agg_idx = AGG_OPS.index(sel_tokens[0].upper())
            col_tok = sel_tokens[1] if len(sel_tokens) > 1 else ""
        else:
            col_tok = sel_tokens[0]

        col_match = re.match(r"<c(\d+)>", col_tok)
        if not col_match:
            return None
        sel_col = int(col_match.group(1))

        conds = []
        if where_part:
            clauses = re.split(r"\s+and\s+", where_part.strip())
            for clause in clauses:
                matched = False
                for op_idx, op_sym in enumerate(COND_OPS):
                    pattern = rf"^(<c\d+>)\s*{re.escape(op_sym)}\s*(.*)$"
                    m = re.match(pattern, clause.strip())
                    if m:
                        c_tok, val = m.group(1), m.group(2).strip()
                        c_idx = int(re.match(r"<c(\d+)>", c_tok).group(1))
                        # Restore original casing if question is supplied
                        if original_question:
                            val = restore_casing(val, original_question)
                        conds.append([c_idx, op_idx, val])
                        matched = True
                        break
                if not matched:
                    return None

        return {"sel": sel_col, "agg": agg_idx, "conds": conds}
    except Exception:
        return None


def format_readable_sql(parsed_query, header):
    if parsed_query is None or "error" in parsed_query:
        return "ERROR: Unparseable SQL"

    q = parsed_query.get("query", parsed_query)
    sel_idx = q["sel"]
    col_name = f'"{header[sel_idx]}"' if sel_idx < len(header) else f"col_{sel_idx}"

    if q["agg"] > 0:
        agg_name = AGG_OPS[q["agg"]]
        select_clause = f"SELECT {agg_name}({col_name}) FROM table"
    else:
        select_clause = f"SELECT {col_name} FROM table"

    if not q["conds"]:
        return select_clause

    where_parts = []
    for col_i, op_i, val in q["conds"]:
        c_name = f'"{header[col_i]}"' if col_i < len(header) else f"col_{col_i}"
        op_str = COND_OPS[op_i]
        where_parts.append(f"{c_name} {op_str} '{val}'")

    return f"{select_clause} WHERE {' AND '.join(where_parts)}"

# """
# decode.py (V2)
# Autoregressive Greedy & Beam Search Decoding with Fuzzy Condition Snapping.
# """

# import re
# import difflib
# import torch
# import torch.nn.functional as F
# from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID

# AGG_OPS = ["", "max", "min", "count", "sum", "avg"]
# COND_OPS = ["=", ">", "<"]


# def snap_condition_value(pred_val: str, original_question: str) -> str:
#     """
#     Snaps generated condition values (e.g. 'butler cc') to the exact
#     matching substring in the question (e.g. 'Butler CC (KS)') to
#     ensure successful casing and punctuation matches in SQLite.
#     """
#     if not original_question or not pred_val:
#         return pred_val

#     cleaned_pred = pred_val.strip().strip("'\"")
#     if not cleaned_pred:
#         return pred_val

#     words = original_question.split()
#     best_candidate = cleaned_pred
#     best_ratio = 0.0

#     # Search all contiguous n-grams in the source question
#     for n in range(1, len(words) + 1):
#         for i in range(len(words) - n + 1):
#             candidate = " ".join(words[i:i + n]).rstrip("?,.!")
#             ratio = difflib.SequenceMatcher(None, cleaned_pred.lower(), candidate.lower()).ratio()
#             if ratio > best_ratio:
#                 best_ratio = ratio
#                 best_candidate = candidate

#     return best_candidate if best_ratio >= 0.60 else cleaned_pred


# def greedy_decode(model, src_ids, max_len=64, device="cpu"):
#     """
#     Greedy autoregressive generation loop.
#     Returns:
#         generated_ids: list of integer token IDs
#         cross_attentions: Tensor of cross-attention weights from the last decoder layer
#     """
#     model.eval()
#     with torch.no_grad():
#         src_ids = src_ids.to(device)
#         enc_mask = model.make_src_mask(src_ids)
#         memory = model.encode(src_ids, enc_mask)

#         ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
#         last_attn = None

#         for _ in range(max_len):
#             tgt_mask = model.make_tgt_mask(ys)
#             out, cross_attns = model.decode(ys, memory, enc_mask, tgt_mask)
#             logits = model.generator(out[:, -1, :])
#             next_token = logits.argmax(dim=-1).item()

#             if cross_attns:
#                 last_attn = cross_attns[-1]

#             ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
#             if next_token == EOS_ID:
#                 break

#         return ys.squeeze(0).tolist(), last_attn


# def beam_search_decode(model, src_ids, beam_size=4, max_len=64, device="cpu"):
#     """
#     Beam search autoregressive generation with length penalty (alpha=0.6).
#     """
#     model.eval()
#     with torch.no_grad():
#         src_ids = src_ids.to(device)
#         enc_mask = model.make_src_mask(src_ids)
#         memory = model.encode(src_ids, enc_mask)

#         # Beam structure: (log_prob, sequence_tensor)
#         beams = [(0.0, torch.tensor([[BOS_ID]], dtype=torch.long, device=device))]
#         completed_beams = []

#         for _ in range(max_len):
#             new_candidates = []
#             all_done = True

#             for score, ys in beams:
#                 if ys[0, -1].item() == EOS_ID:
#                     completed_beams.append((score, ys))
#                     continue

#                 all_done = False
#                 tgt_mask = model.make_tgt_mask(ys)
#                 out, _ = model.decode(ys, memory, enc_mask, tgt_mask)
#                 log_probs = F.log_softmax(model.generator(out[:, -1, :]), dim=-1)

#                 topk_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

#                 for k in range(beam_size):
#                     next_token = topk_indices[0, k].item()
#                     cand_score = score + topk_probs[0, k].item()
#                     cand_ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
#                     new_candidates.append((cand_score, cand_ys))

#             if all_done or not new_candidates:
#                 break

#             # Sort and select top beams with length penalty
#             def length_penalty(seq_len, alpha=0.6):
#                 return ((5.0 + seq_len) / 6.0) ** alpha

#             new_candidates.sort(key=lambda x: x[0] / length_penalty(x[1].size(1)), reverse=True)
#             beams = new_candidates[:beam_size]

#         if not completed_beams:
#             completed_beams = beams

#         completed_beams.sort(key=lambda x: x[0] / length_penalty(x[1].size(1)), reverse=True)
#         best_seq = completed_beams[0][1].squeeze(0).tolist()
#         return best_seq, None


# def parse_sql_string(raw_sql: str, original_question: str = None) -> dict:
#     """
#     Converts serialized SQL string tokens into an official WikiSQL query dict:
#     {"sel": int, "agg": int, "conds": [[col, op, val], ...]}
#     """
#     text = raw_sql.strip().lower()
#     if not text.startswith("select"):
#         return None

#     # Strip 'select'
#     remainder = text[len("select"):].strip()

#     # Split on 'where'
#     if "where" in remainder:
#         select_part, where_part = remainder.split("where", 1)
#     else:
#         select_part, where_part = remainder, None

#     # Parse Aggregation & Select Column
#     select_tokens = select_part.strip().split()
#     if not select_tokens:
#         return None

#     agg_idx = 0
#     col_str = None

#     if len(select_tokens) == 1:
#         col_str = select_tokens[0]
#     elif len(select_tokens) >= 2:
#         if select_tokens[0] in AGG_OPS:
#             agg_idx = AGG_OPS.index(select_tokens[0])
#             col_str = select_tokens[1]
#         else:
#             col_str = select_tokens[0]

#     # Extract column index: <c{idx}>
#     col_match = re.search(r"<c(\d+)>", col_str) if col_str else None
#     if not col_match:
#         return None
#     sel_idx = int(col_match.group(1))

#     # Parse WHERE conditions
#     conds = []
#     if where_part:
#         raw_conds = where_part.split(" and ")
#         for raw_c in raw_conds:
#             c = raw_c.strip()
#             # Match: <c(\d+)> (=|>|<) (value)
#             cond_match = re.search(r"<c(\d+)>\s*([=><])\s*(.*)", c)
#             if cond_match:
#                 c_idx = int(cond_match.group(1))
#                 op_sym = cond_match.group(2)
#                 val = cond_match.group(3).strip()

#                 # Snap value to exact question text if original_question is provided
#                 val = snap_condition_value(val, original_question)

#                 op_idx = COND_OPS.index(op_sym) if op_sym in COND_OPS else 0
#                 conds.append([c_idx, op_idx, val])

#     return {
#         "sel": sel_idx,
#         "agg": agg_idx,
#         "conds": conds
#     }


# def format_readable_sql(sql_dict: dict, header: list) -> str:
#     """
#     Reconstructs an executable SQLite query string with real header column names.
#     """
#     if not sql_dict:
#         return ""

#     sel_idx = sql_dict["sel"]
#     col_name = f'"{header[sel_idx]}"' if sel_idx < len(header) else f'"col_{sel_idx}"'
#     agg_op = AGG_OPS[sql_dict["agg"]].upper() if sql_dict["agg"] < len(AGG_OPS) else ""

#     select_clause = f"{agg_op}({col_name})" if agg_op else col_name
#     query = f"SELECT {select_clause} FROM table"

#     if sql_dict.get("conds"):
#         cond_strings = []
#         for c_idx, op_idx, val in sql_dict["conds"]:
#             c_name = f'"{header[c_idx]}"' if c_idx < len(header) else f'"col_{c_idx}"'
#             op_sym = COND_OPS[op_idx] if op_idx < len(COND_OPS) else "="
            
#             # Format numeric vs string literals
#             try:
#                 float(val)
#                 cond_strings.append(f"{c_name} {op_sym} {val}")
#             except ValueError:
#                 cond_strings.append(f"{c_name} {op_sym} '{val}'")
                
#         query += " WHERE " + " AND ".join(cond_strings)

#     return query