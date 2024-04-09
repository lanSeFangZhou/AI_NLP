import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# 基本组件
# 2.1 MultiHeadAttention
# 自注意力的attn_mask可通过如下代码快速生成

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)

    def forward(self, X, attn_mask=None, key_padding_mask=None):
        '''
        :param X: (L, N, embed_dim)
        :param attn_mask:
        :param key_padding_mask:
        :return:
        '''
        return self.mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        '''
        :param query: (n, N, embed_dim)
        :param key: (m, N, embed_dim)
        :param value: (m, N, embed_dim)
        :param attn_mask: (n, m) or (N * num_heads, n, m)
        :param key_padding_mask: (N, m)
        :return:
            attn_output: (n, N, embed_dim)
            attn_output_weights: (N, num_heads, n, m)
        '''
        return self._multi_head_attention_forward(query,
                                                  key,
                                                  value,
                                                  dropout_p=self.dropout,
                                                  attn_mask=attn_mask,
                                                  key_padding_mask=key_padding_mask,
                                                  training=self.training)

    def _multi_head_attention_forward(self, query, key, value, dropout_p, attn_mask=None, key_padding_mask=None, training=True):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        n, N, embed_dim = q.size()
        m = key.size(0)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                assert attn_mask.shape == (n, m)
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                assert attn_mask.shape == (N * self.num_heads, n, m)
            else:
                raise RuntimeError

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (N, m)
            key_padding_mask = key_padding_mask.view(N, 1, 1, m).repeat(1, self.num_heads, 1, 1).reshape(N * self.num_heads, 1, m)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, -1e9)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, -1e9)
            attn_mask = new_attn_mask

        q = q.reshape(n, N * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1)

        if not training:
            dropout_p = 0.0

        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).reshape(n, N, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output_weights = attn_output_weights.reshape(N, self.num_heads, n, m)
        return attn_output, attn_output_weights

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0):
        '''
        :param q: (N, n, E)
        :param k: (N, m, E)
        :param v: (N, m, E)
        :param attn_mask: (n, m) or (N, n, m)
        :param dropout_p:
        :return:
            attn_output: (N, n, E)
            attn_weights: (N, n, m)
        '''
        q = q / math.sqrt(q.size(2))
        if attn_mask is not None:
            scores = q @ k.transpose(-2, -1) + attn_mask
        else:
            scores = q @ k.transpose(-2, -1)

        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        attn_output = attn_weights @ v
        return attn_output, attn_weights

def generate_square_subsequent_mask(a):
    return torch.triu(torch.full((a, a), -1e9), diagonal=1)

# key_padding_mask, 以源序列src为例，初始输入形状为(N, S),设<pad>在词表中的索引为0，则：
# 一个可能的例子
src = torch.tensor([
    [3, 5, 7, 0, 0],
    [9, 4, 0, 0, 0],
    [6, 7, 2, 1, 0],
])
src_key_padding_mask = src == 0
print(src_key_padding_mask)

# 2.2 PositionalEncoding
class PositionalEmbedding(nn.Module):
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

# 2.3 PositionWiseFFN, 一个隐藏层的MLP
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

# 2.4 AddNorm: 在NLP任务中，BatchNorm的效果通常没有LayerNorm的效果好，所以我们在残差连接后接上LayerNorm
class AddNorm(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.norm(X + self.dropout(Y))

# 搭建transformer
# 3.1 Encoder
# 首先需要实现一个TransformerEncoderLayer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        X = src
        X = self.addnorm1(X, self.self_attn(X, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0])
        X = self.addnorm2(X, self.ffn(X))
        return X

# 为了将多个EncoderLayer组合在一起形成Encoder，我们需要定义一个复制layer的函数
# 将module复制N次
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 接下来实现Encoder
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
    # 我们实现的Encoder并不自带位置编码(后续的Decoder也是如此)，这样做是为了在面对不同任务时，我们不需要改动太多的代码
    # Encoder最后一层的输出称为Memory

# 3.2 Decoder
# 先实现一个DecoderLayer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        X = tgt
        X = self.addnorm1(X, self.self_attn(X, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        X = self.addnorm2(X, self.cross_attn(X, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0])
        X = self.addnorm3(X, self.ffn(X))
        return X

# 然后进行组装
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
    # 有一点需要注意，我们实现的Decoder不包含最后一个Linear层

# 3.3 Transformer
# 有了Encoder和Decoder之后，我们就可以组装Transformer了
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
                memory_key_padding_mask=None):
        '''
        :param src: (S, N, E)
        :param tgt: (T, N, E)
        :param src_mask: (S, S) or (N * num_heads, S, S)
        :param tgt_mask: (T, T) or (N * num_heads, T, T)
        :param memory_mask: (T, S)
        :param src_key_padding_mask: (N, S)
        :param tgt_key_padding_mask: (N, T)
        :param memory_key_padding_mask: (N, S)
        :return:
            output: (T, N, E)
        '''
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, a):
        return torch.triu(torch.full((a, a), -1e9), diagonal=1)

    def _reset_parameters(self):
        '''Initiate parammeters in the transformer model'''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# 3.4 验证
src_len = 5
tgt_len = 6
batch_size = 2
d_model = 16
nhead = 8

src = torch.randn(src_len, batch_size, d_model)
tgt = torch.randn(tgt_len, batch_size, d_model)

src_key_padding_mask = torch.tensor([[False, False, False,  True, True],
                                     [False, False, False, False, True]])
tgt_key_padding_mask = torch.tensor([[False, False, False,  True, True, True],
                                     [False, False, False, False, True, True]])

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













