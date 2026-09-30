import re
import json
import torch
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from starter.data_prep import AGG_OPS, COND_OPS
from model.transformer import make_padding_mask, make_causal_mask


def greedy_decode(model, src, max_len=64, device="cpu"):
    """
    Greedy decoding: selects token with highest probability at each step.
    src: (1, S)
    """
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
    """
    Beam search decoding with beam_size=4, stopping at EOS_ID or max_len.
    src: (1, S)
    """
    model.eval()
    src = src.to(device)
    src_mask = make_padding_mask(src, PAD_ID)

    with torch.no_grad():
        memory = model.encode(src, src_mask=src_mask)
        
        # Candidate tuples: (token_list, cumulative_log_prob)
        beams = [([BOS_ID], 0.0)]
        completed_beams = []

        for _ in range(max_len):
            new_candidates = []
            for tokens, score in beams:
                if tokens[-1] == EOS_ID:
                    completed_beams.append((tokens, score))
                    continue

                ys = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_mask = make_causal_mask(ys.size(1), device=device)
                out, _ = model.decode(ys, memory, tgt_mask=tgt_mask, cross_mask=src_mask)
                log_probs = torch.log_softmax(model.generator(out[:, -1, :]), dim=-1).squeeze(0)

                topk_probs, topk_indices = torch.topk(log_probs, beam_size)
                for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                    new_candidates.append((tokens + [idx], score + prob))

            if not new_candidates:
                break

            # Retain top beam_size hypotheses sorted by score
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = new_candidates[:beam_size]

            # If all active beams have finished
            if all(b[0][-1] == EOS_ID for b in beams):
                completed_beams.extend(beams)
                break

        if not completed_beams:
            completed_beams = beams

        completed_beams.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
        best_tokens = completed_beams[0][0]

    return best_tokens, None


def parse_sql_string(sql_str):
    """
    Parses predicted string (e.g., 'select count <c2> where <c0> = kim manners')
    into WikiSQL dict format:
    {"sel": int, "agg": int, "conds": [[col, op, value], ...]}
    Returns None if malformed.
    """
    try:
        text = sql_str.strip().lower()
        if not text.startswith("select "):
            return None

        # Split SELECT clause and WHERE clause
        if " where " in text:
            select_part, where_part = text.split(" where ", 1)
        else:
            select_part, where_part = text, ""

        sel_tokens = select_part.replace("select", "").strip().split()
        if not sel_tokens:
            return None

        agg_idx = 0
        sel_col = None

        # Check for aggregation
        agg_candidates = [op.lower() for op in AGG_OPS[1:]]  # max, min, count, sum, avg
        if sel_tokens[0] in agg_candidates:
            agg_idx = AGG_OPS.index(sel_tokens[0].upper())
            col_token = sel_tokens[1] if len(sel_tokens) > 1 else ""
        else:
            col_token = sel_tokens[0]

        # Extract column index <c{i}>
        col_match = re.match(r"<c(\d+)>", col_token)
        if not col_match:
            return None
        sel_col = int(col_match.group(1))

        # Parse conditions in WHERE clause
        conds = []
        if where_part:
            # Clauses joined by ' and '
            clauses = re.split(r"\s+and\s+", where_part.strip())
            for clause in clauses:
                matched = False
                for op_idx, op_sym in enumerate(COND_OPS):
                    pattern = rf"^(<c\d+>)\s*{re.escape(op_sym)}\s*(.*)$"
                    m = re.match(pattern, clause.strip())
                    if m:
                        c_tok, val = m.group(1), m.group(2).strip()
                        c_idx = int(re.match(r"<c(\d+)>", c_tok).group(1))
                        conds.append([c_idx, op_idx, val])
                        matched = True
                        break
                if not matched:
                    return None

        return {"sel": sel_col, "agg": agg_idx, "conds": conds}
    except Exception:
        return None


def format_readable_sql(parsed_query, header):
    """
    Converts parsed WikiSQL JSON representation into human-readable SQL text
    using actual table header column names.
    """
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

    where_clauses = []
    for col_i, op_i, val in q["conds"]:
        c_name = f'"{header[col_i]}"' if col_i < len(header) else f"col_{col_i}"
        op_str = COND_OPS[op_i]
        where_clauses.append(f"{c_name} {op_str} '{val}'")

    return f"{select_clause} WHERE {' AND '.join(where_clauses)}"