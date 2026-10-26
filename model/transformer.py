import torch
import torch.nn as nn

PAD_ID = 0

def make_pad_mask(ids, pad_id=PAD_ID):
    mask = ids != pad_id

    return mask.unsqueeze(1).unsqueeze(2)

def make_causal_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask

def make_tgt_mask(tgt_ids):
    pad_mask = make_pad_mask(tgt_ids)                 
    causal_mask = make_causal_mask(tgt_ids.size(1)).to(tgt_ids.device)  
    mask = pad_mask & causal_mask
    return mask

from model.layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers=3, d_model=256, num_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        layer_list = []
        for i in range(num_layers):
            layer_list.append(EncoderLayer(d_model, num_heads, d_ff, dropout))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

from model.layers import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, num_layers=3, d_model=256, num_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        layer_list = []
        for i in range(num_layers):
            layer_list.append(DecoderLayer(d_model, num_heads, d_ff, dropout))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
            
        return x

from starter.embeddings import TokenEmbedding, InputLayer

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=3,
                 d_ff=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id

        shared = TokenEmbedding(vocab_size, d_model, pad_id)
        self.src_input = InputLayer(shared, d_model, dropout=dropout)
        self.tgt_input = InputLayer(shared, d_model, dropout=dropout)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)
        self.output_layer.weight = shared.emb.weight

        # small starting values, so the first scores are not huge
        nn.init.normal_(shared.emb.weight, mean=0.0, std=d_model ** -0.5)
        shared.emb.weight.data[pad_id].zero_()

    def encode(self, src):
        src_mask = make_pad_mask(src, self.pad_id)
        x = self.src_input(src)
        enc_out = self.encoder(x, src_mask)
        return enc_out, src_mask

    def decode(self, tgt_in, enc_out, src_mask):
        tgt_mask = make_tgt_mask(tgt_in)
        x = self.tgt_input(tgt_in)
        x = self.decoder(x, enc_out, src_mask, tgt_mask)
        return self.output_layer(x)

    def forward(self, src, tgt_in):
        enc_out, src_mask = self.encode(src)
        return self.decode(tgt_in, enc_out, src_mask)
