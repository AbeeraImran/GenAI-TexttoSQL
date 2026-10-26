import sys
sys.path.append("starter")
import torch
import torch.nn as nn

PAD_ID = 0

def make_loss_function():
    return nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

def compute_loss(model, loss_fn, src, tgt):
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    logits = model(src, tgt_in)          
    vocab_size = logits.size(-1)
    
    return loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

def get_lr(step, d_model=256, warmup_steps=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def train_one_epoch(model, loader, loss_fn, optimizer, step, device):
    model.train()          
    total_loss = 0
    num_batches = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        step = step + 1
        set_lr(optimizer, get_lr(step))

        optimizer.zero_grad()
        loss = compute_loss(model, loss_fn, src, tgt)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
        num_batches = num_batches + 1

    return total_loss / num_batches, step

def evaluate(model, loader, loss_fn, device):
    model.eval()          
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            loss = compute_loss(model, loss_fn, src, tgt)
            total_loss = total_loss + loss.item()
            num_batches = num_batches + 1

    return total_loss / num_batches

import os
import json
import time
import sentencepiece as spm
from dataset import make_loader
from model.transformer import Transformer

CHECKPOINT_DIR = "checkpoints"

def main():
    num_epochs = 20
    if len(sys.argv) > 1:
        num_epochs = int(sys.argv[1])   

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print("device:", device, gpu_name)

    sp = spm.SentencePieceProcessor(model_file="sql_sp.model")
    train_dl = make_loader("train_pairs.jsonl", sp, train=True, batch_size=64)
    dev_dl = make_loader("dev_pairs.jsonl", sp, train=False, batch_size=64)

    model = Transformer(vocab_size=sp.get_piece_size()).to(device)
    loss_fn = make_loss_function()
    optimizer = make_optimizer(model)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    last_path = CHECKPOINT_DIR + "/last.pt"
    best_path = CHECKPOINT_DIR + "/best.pt"

    start_epoch = 1
    step = 0
    best_dev_loss = 1e9
    history = []

    if os.path.exists(last_path):
        saved = torch.load(last_path, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        step = saved["step"]
        best_dev_loss = saved["best_dev_loss"]
        history = saved["history"]
        print("resuming after epoch", saved["epoch"])

    for epoch in range(start_epoch, num_epochs + 1):
        start = time.time()

        train_loss, step = train_one_epoch(model, train_dl, loss_fn, optimizer, step, device)
        dev_loss = evaluate(model, dev_dl, loss_fn, device)

        seconds = time.time() - start
        lr_now = get_lr(step)
        print("epoch", epoch, "| train loss", round(train_loss, 4), "| dev loss", round(dev_loss, 4),
              "| lr", lr_now, "| seconds", round(seconds))
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                        "lr": lr_now, "seconds": seconds})

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), best_path)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "step": step, "best_dev_loss": best_dev_loss,
                    "history": history}, last_path)
        with open("results/training_log.json", "w") as f:
            json.dump({"gpu": gpu_name, "history": history}, f)

if __name__ == "__main__":
    main()
