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