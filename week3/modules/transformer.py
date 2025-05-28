
import torch.nn as nn
from .positional import PositionalEncoding
from .decoder import DecoderLayer
from .encoder import EncoderLayer

# 7. Full Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, d_ff=2048, num_layers=6, max_len=100):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_emb(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos_enc(self.tgt_emb(tgt))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)
print(dir)