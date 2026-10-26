import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import gradio as gr
import sentencepiece as spm
from model.transformer import Transformer
from decode import greedy_decode, ids_to_text, parse_sql, to_readable_sql
from starter.data_prep import encode_source

EOS = 3
MAX_COLS = 64

sp = spm.SentencePieceProcessor()
sp.load("sql_sp.model")

model = Transformer(8000)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()


def generate_sql(question, columns_text):
    header = [c.strip() for c in columns_text.split(",") if c.strip()]
    if question.strip() == "" or len(header) == 0:
        return "Please type a question and at least one column name.", ""
    if len(header) > MAX_COLS:
        return "Too many columns (the model supports up to 64).", ""

    source_text = encode_source(question, header)

    ids = sp.encode(source_text) + [EOS]
    src = torch.tensor([ids])
    out = greedy_decode(model, src)[0].tolist()
    raw = ids_to_text(out[1:], sp)

    parsed = parse_sql(raw, num_cols=len(header))

    sql = to_readable_sql(parsed, header)
    return sql, raw


demo = gr.Interface(
    fn=generate_sql,
    inputs=[
        gr.Textbox(label="Question", placeholder="What is Terrence Ross' nationality?"),
        gr.Textbox(label="Column names (comma-separated)",
                   placeholder="Player, No., Nationality, Position, Years in Toronto, School/Club Team"),
    ],
    outputs=[gr.Textbox(label="Generated SQL"), gr.Textbox(label="Raw model output")],
    title="Text-to-SQL Transformer",
)

if __name__ == "__main__":
    demo.launch(share=True)
