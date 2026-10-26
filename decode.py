
import torch
from model.transformer import Transformer

PAD, BOS, EOS = 0, 2, 3
MAX_LEN = 64


def greedy_decode(model, src, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src)        
        batch_size = src.size(0)

        tgt = torch.full((batch_size, 1), BOS, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for step in range(max_len):
            logits = model.decode(tgt, enc_out, src_mask)  

            last_logits = logits[:, -1, :]

            next_token = last_logits.argmax(dim=-1)

            next_token[finished] = PAD                      
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == EOS)

            if finished.all():
                break
    return tgt                                              


def beam_search(model, src, beam_size=4, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src)            

        beams = [([BOS], 0.0)]                           
        finished = []

        for step in range(max_len):
            candidates = []
            for tokens, score in beams:
                tgt = torch.tensor([tokens], device=src.device)
                logits = model.decode(tgt, enc_out, src_mask)

                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)

                top_scores, top_ids = log_probs.topk(beam_size)
                for s, t in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append((tokens + [t], score + s))

            candidates.sort(key=lambda c: c[1], reverse=True)

            beams = []
            for tokens, score in candidates[:beam_size]:
                if tokens[-1] == EOS:
                    finished.append((tokens, score))
                else:
                    beams.append((tokens, score))

            if len(beams) == 0 or len(finished) >= beam_size:
                break

        pool = finished + beams                       
        best_tokens, best_score = max(pool, key=lambda c: c[1])

        best_tokens = best_tokens[1:]                  
        if EOS in best_tokens:
            best_tokens = best_tokens[:best_tokens.index(EOS)]
        return best_tokens                             


from starter.data_prep import AGG_OPS, COND_OPS


def is_col(token):
    # True for tokens like <c0>, <c12>
    return token.startswith("<c") and token.endswith(">") and token[2:-1].isdigit()


def col_index(token):
    return int(token[2:-1])


def parse_sql(text, num_cols=None):
    tokens = text.split()
    agg_names = [a.lower() for a in AGG_OPS]    

    if len(tokens) == 0 or tokens[0] != "select":
        return None
    pos = 1

    agg = 0
    if pos < len(tokens) and tokens[pos] in agg_names[1:]:
        agg = agg_names.index(tokens[pos])
        pos += 1

    if pos >= len(tokens) or not is_col(tokens[pos]):
        return None
    sel = col_index(tokens[pos])
    pos += 1

    conds = []
    while pos < len(tokens):
        keyword = "where" if len(conds) == 0 else "and"
        if tokens[pos] != keyword:
            return None
        pos += 1

        if pos >= len(tokens) or not is_col(tokens[pos]):
            return None
        col = col_index(tokens[pos])
        pos += 1

        if pos >= len(tokens) or tokens[pos] not in COND_OPS:
            return None
        op = COND_OPS.index(tokens[pos])
        pos += 1

        value_tokens = []
        while pos < len(tokens):
            if tokens[pos] == "and" and pos + 1 < len(tokens) and is_col(tokens[pos + 1]):
                break
            value_tokens.append(tokens[pos])
            pos += 1

        if len(value_tokens) == 0:
            return None
        conds.append([col, op, " ".join(value_tokens)])

    if num_cols is not None:
        if sel >= num_cols or any(c[0] >= num_cols for c in conds):
            return None

    return {"sel": sel, "agg": agg, "conds": conds}


import json
import sys
sys.path.insert(0, "starter")
from dataset import make_loader


def ids_to_text(ids, sp):
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    ids = [i for i in ids if i != PAD]
    return sp.decode(ids)


def decode_split_greedy(model, sp, pairs_path, out_path, device, batch_size=64, max_batches=None):
    model.to(device)
    loader = make_loader(pairs_path, sp, train=False, batch_size=batch_size)
    n_lines = 0
    n_failed = 0

    with open(out_path, "w") as f_out:
        for b, (src, tgt) in enumerate(loader):
            if max_batches is not None and b == max_batches:
                break
            src = src.to(device)
            out = greedy_decode(model, src)              

            for row in out.tolist():
                text = ids_to_text(row[1:], sp)

                parsed = parse_sql(text)
                if parsed is None:
                    n_failed += 1
                    line = {"error": "parse"}
                else:
                    line = {"query": parsed}
                f_out.write(json.dumps(line) + "\n")
                n_lines += 1

    return n_lines, n_failed


def beam_search_batched(model, src, beam_size=4, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src)           

        beams = [([BOS], 0.0)]
        finished = []

        for step in range(max_len):
            n = len(beams)

            enc_n = enc_out.expand(n, -1, -1)
            mask_n = src_mask.expand(n, -1, -1, -1)

            tgt = torch.tensor([tokens for tokens, score in beams], device=src.device)
            logits = model.decode(tgt, enc_n, mask_n)           
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1) 

            candidates = []
            for i in range(n):
                tokens, score = beams[i]
                top_scores, top_ids = log_probs[i].topk(beam_size)
                for s, t in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append((tokens + [t], score + s))

            candidates.sort(key=lambda c: c[1], reverse=True)

            beams = []
            for tokens, score in candidates[:beam_size]:
                if tokens[-1] == EOS:
                    finished.append((tokens, score))
                else:
                    beams.append((tokens, score))

            if len(beams) == 0 or len(finished) >= beam_size:
                break

        pool = finished + beams
        best_tokens, best_score = max(pool, key=lambda c: c[1])
        best_tokens = best_tokens[1:]
        if EOS in best_tokens:
            best_tokens = best_tokens[:best_tokens.index(EOS)]
        return best_tokens


def decode_split_beam(model, sp, pairs_path, out_path, device, beam_size=4, max_examples=None):
    model.to(device)
    model.eval()
    loader = make_loader(pairs_path, sp, train=False, batch_size=1)  
    n_lines = 0
    n_failed = 0

    with open(out_path, "w") as f_out:
        for i, (src, tgt) in enumerate(loader):
            if max_examples is not None and i == max_examples:
                break
            ids = beam_search_batched(model, src.to(device), beam_size=beam_size)
            text = ids_to_text(ids, sp)
            parsed = parse_sql(text)
            if parsed is None:
                n_failed += 1
                line = {"error": "parse"}
            else:
                line = {"query": parsed}
            f_out.write(json.dumps(line) + "\n")
            n_lines += 1
    return n_lines, n_failed


def column_name(header, index):
    if index < len(header):
        return header[index]
    return "col" + str(index)         


def to_readable_sql(query, header, table_name="table"):
    if query is None:
        return "(parse error)"

    sel_name = column_name(header, query["sel"])

    if query["agg"] != 0:
        select_part = AGG_OPS[query["agg"]] + "(" + sel_name + ")"
    else:
        select_part = sel_name

    sql = "SELECT " + select_part + " FROM " + table_name

    parts = []
    for col, op, value in query["conds"]:
        parts.append(column_name(header, col) + " " + COND_OPS[op] + " '" + str(value) + "'")

    if len(parts) > 0:
        sql = sql + " WHERE " + " AND ".join(parts)
    return sql
