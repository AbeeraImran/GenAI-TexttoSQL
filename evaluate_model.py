import json
import os
import sys

import torch
import sentencepiece as spm
import matplotlib.pyplot as plt

from model.transformer import Transformer
from decode import decode_split_greedy, decode_split_beam, greedy_decode, ids_to_text
from starter.data_prep import load_split

EOS = 3


def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor()
    sp.load("sql_sp.model")
    model = Transformer(8000)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.to(device)
    model.eval()
    return model, sp, device


def write_predictions(what, model, sp, device):
    split, mode = what.split("_")                     
    pairs_path = split + "_pairs.jsonl"
    out_path = "results/pred_" + split + "_" + mode + ".jsonl"

    if split == "test" and os.path.exists(out_path):
        print("STOP: test predictions already exist. The test set is used once only.")
        return

    if mode == "greedy":
        n_lines, n_failed = decode_split_greedy(model, sp, pairs_path, out_path, device)
    else:
        n_lines, n_failed = decode_split_beam(model, sp, pairs_path, out_path, device, beam_size=4)

    print("lines written:", n_lines)
    print("parse failures:", n_failed, "=", round(100 * n_failed / n_lines, 2), "%")
    print("now run the official evaluator:")
    print("python WikiSQL/evaluate.py WikiSQL/data/" + split + ".jsonl WikiSQL/data/" + split + ".db " + out_path)


def clean_conds(conds):
    result = set()
    for col, op, value in conds:
        result.add((col, op, str(value).lower().strip()))
    return result


def component_accuracy(pred_path):
    examples, tables = load_split("dev")
    with open(pred_path) as f:
        preds = [json.loads(line) for line in f]

    sel_ok = 0
    agg_ok = 0
    where_ok = 0
    all_ok = 0

    for ex, pred in zip(examples, preds):
        if "query" not in pred:
            continue
        gold = ex["sql"]
        q = pred["query"]

        s = q["sel"] == gold["sel"]
        a = q["agg"] == gold["agg"]
        w = clean_conds(q["conds"]) == clean_conds(gold["conds"])

        sel_ok += s
        agg_ok += a
        where_ok += w
        all_ok += (s and a and w)

    n = len(examples)
    print(pred_path)
    print("  SELECT column accuracy:", round(sel_ok / n, 4))
    print("  AGG accuracy          :", round(agg_ok / n, 4))
    print("  WHERE (set) accuracy  :", round(where_ok / n, 4))
    print("  all three correct     :", round(all_ok / n, 4))


def attention_map(model, sp, device, example_index=1):
    with open("dev_pairs.jsonl") as f:
        for i, line in enumerate(f):
            if i == example_index:
                pair = json.loads(line)
                break

    src_ids = sp.encode(pair["src"]) + [EOS]
    src = torch.tensor([src_ids]).to(device)

    out = greedy_decode(model, src)[0].tolist()
    if EOS in out:
        out = out[:out.index(EOS) + 1]
    tgt = torch.tensor([out]).to(device)

    with torch.no_grad():
        enc_out, src_mask = model.encode(src)
        model.decode(tgt, enc_out, src_mask)

    weights = model.decoder.layers[-1].cross_attn.last_weights
    attn = weights.mean(dim=1)[0]

    print("attention shape:", tuple(attn.shape))
    print("row sums (should be about 1):", attn.sum(dim=-1)[:3])

    src_labels = [sp.id_to_piece(i) for i in src_ids]
    tgt_labels = [sp.id_to_piece(i) for i in out]
    plt.figure(figsize=(14, 4))
    plt.imshow(attn.cpu().numpy(), aspect="auto")
    plt.xticks(range(len(src_labels)), src_labels, rotation=90, fontsize=7)
    plt.yticks(range(len(tgt_labels)), tgt_labels, fontsize=8)
    plt.xlabel("source tokens")
    plt.ylabel("generated tokens")
    plt.title("Cross-attention (last decoder layer, mean over heads)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/fig_cross_attention.png", dpi=150)
    print("generated:", ids_to_text(out[1:], sp))


if __name__ == "__main__":
    usage = "usage: python evaluate_model.py [dev_greedy | dev_beam | test_greedy | components | attention]"
    if len(sys.argv) < 2:
        print(usage)
        sys.exit()

    what = sys.argv[1]
    if what == "components":
        component_accuracy("results/pred_dev_greedy.jsonl")
        component_accuracy("results/pred_dev_beam.jsonl")
    elif what in ("dev_greedy", "dev_beam", "test_greedy"):
        model, sp, device = load_model_and_tokenizer()
        write_predictions(what, model, sp, device)
    elif what == "attention":
        model, sp, device = load_model_and_tokenizer()
        attention_map(model, sp, device)
    else:
        print(usage)
