import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# MultiHeadAttention
# 生成自注意力中的attn_mask
def generate_squre_subsequent_mask(a):
    return torch.triu(torch.full((a, a), -1e9), diagonal=1)

'''一个可能的例子'''
src = torch.tensor(
    [
        [3, 5, 7, 0, 0],
        [9, 4, 0, 0, 0],
        [6, 7, 2, 1, 0],
    ]
)
src_key_padding_mask = src == 0
print(src_key_padding_mask)


# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros(max_len, d_model)
        row = torch.arange(max_len).reshape(-1, 1)
        col = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        self.P[:, ::2] = torch.sin(row / col)
        self.P[:, 1::2] = torch.cos(row / col)
        self.P = self.P.unsqueeze(0).transpose(0, 1)

    def forward(self, X):
        X = X + self.P[:X.shape[0]].to(X.device)
        return self.dropout(X)

# PositionWiseFFN
class FFN(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, X):
        return self.net(X)

# AddNorm
class AddNorm(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.norm(X + self.dropout(Y))

# 搭建transformer
# encoder
# transformerEncoderLayer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nHead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nHead, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        X = src
        X = self.addnorm1(X, self.self_attn(X, attn_mask=src_mask, key_padding_mask=src_key_padding_mask))
        X = self.addnorm2(X, self.ffn(X))
        return X

# 将module复制N次
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=6, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

# Decoder
# DecoderLayer
class TransformerDecoderLayer(nn.Module):
    def __init_(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_paddding_mask=None, memory_key_padding_mask=None):
        X = tgt
        X = self.addnorm1(X, self.self_attn(X, attn_mask=tgt_mask, key_padding_mask=tgt_key_paddding_mask)[0])
        X = self.addnorm2(X, self.cross_attn(X, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0])
        X = self.addnorm3(X, self.ffn(X))
        return X

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers=6, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

# Transformer
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def forward(self,
                src,
                tgt,
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
                ):
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, a):
        return torch.triu(torch.full((a, a), -1e9), diagonal=1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# 沿验证
src_len = 5
tgt_len = 6
batch_size = 2
d_model = 16
nhead= 18

src = torch.randn(src_len, batch_size, d_model)
tgt = torch.randn(tgt_len, batch_size, d_model)

src_key_padding_mask = torch.Tensor(
    [
        [False, False, False, True, True],
        [False, False, False, False, True]
    ]
)
tgt_key_padding_mask = torch.Tensor([
    [False, False, False, True, True, True],
    [False, False, False, False, True, True]
])

transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=200)

src_mask = transformer.generate_square_subsequent_mask(src_len)
tgt_mask = transformer.generate_square_subsequent_mask(tgt_len)
memory_mask = torch.randint(2, (tgt_len, src_len)) == torch.randint(2, (tgt_len, src_len))

output = transformer(
    src=src,
    tgt=tgt,
    src_mask=src_mask,
    tgt_mask=tgt_mask,
    memory_mask=memory_mask,
    src_key_padding_mask=src_key_padding_mask,
    tgt_key_padding_mask=tgt_key_padding_mask,
    memory_key_padding_mask=src_key_padding_mask
)
print(output.shape)






