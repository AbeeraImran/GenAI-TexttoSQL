import torch
from model.transformer import Transformer

torch.manual_seed(0)
model = Transformer(8000)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()

src = torch.randint(4, 8000, (1, 10))
tgt = torch.randint(4, 8000, (1, 8))

with torch.no_grad():
    enc_out, src_mask = model.encode(src)
    out_a = model.decode(tgt, enc_out, src_mask)

    tgt_changed = tgt.clone()
    tgt_changed[0, -1] = 5 if tgt[0, -1] != 5 else 6
    out_b = model.decode(tgt_changed, enc_out, src_mask)

    earlier_diff = (out_a[:, :-1] - out_b[:, :-1]).abs().max()
    last_diff = (out_a[:, -1] - out_b[:, -1]).abs().max()
    print("1. causal mask")
    print("   difference at earlier positions:", earlier_diff.item(), "(must be 0.0)")
    print("   difference at the last position:", last_diff.item(), "(must be clearly above 0)")

    pad = torch.zeros((1, 6), dtype=torch.long)
    src_padded = torch.cat([src, pad], dim=1)
    enc_p, mask_p = model.encode(src_padded)
    out_c = model.decode(tgt, enc_p, mask_p)
    print("2. padding mask")
    print("   max output difference:", (out_a - out_c).abs().max().item(), "(must be 0.0 or about 1e-6)")

    w = model.encoder.layers[0].self_attn.last_weights          
    sums = w.sum(dim=-1)
    print("3. attention rows")
    print("   all rows sum to 1:", torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
    print("   largest weight on the 6 <pad> columns:", w[..., 10:].max().item(), "(must be about 0)")

same_src = model.src_input.tok.emb.weight is model.output_layer.weight
same_tgt = model.tgt_input.tok.emb.weight is model.output_layer.weight
print("4. weight sharing")
print("   encoder embedding is output weight:", same_src)
print("   decoder embedding is output weight:", same_tgt)
