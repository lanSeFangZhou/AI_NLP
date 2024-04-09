# 各种注意力机制的Pytorch实现

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意力评分函数
# 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, key, value):
        '''
        :param query: (N, n, d_q)
        :param key: (N, m, d_k)
        :param value: (N, m, d_v)
        :return:
        '''
        query, key = self.W_q(query).unsqueeze(2), self.W_k(key).unsqueeze(1)
        attn_weights = F.softmax(self.W_v(torch.tanh(query+key)).sequeeze(), dim=-1) # (N, n, m)
        return attn_weights @ value # (N, n, d_v)
    # @ 相当于 torch.bmm


# 缩放点积注意力
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        '''
        :param query: (N, n, d)
        :param key: (N, m, d)
        :param value: (N, m, d_v)
        :return:
        '''
        return F.softmax(query @ key.transpose(1, 2) / math.sqrt(query.size(2)), dim=-1) @ value

# mask与dropout
a = torch.randn(4, 4)
print(a)
mask = torch.Tensor([
    [False, False, False, True],
    [False, False,  True, True],
    [False,  True,  True, True],
    [True,   True,  True, True]
]) # mask不一定要与a的形状相同，只要能广播成a的形状即可
b = a.masked_fill(mask, 0)
print(b)

# 在引入mask和dropout之后，两种注意力评分函数变为：
class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size, dropout=0):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        '''
        :param query: (N, n, d_q)
        :param key: (N, m, d_k)
        :param value: (N, m, d_v)
        :param attn_mask: (N, n, m)
        :return:
        '''
        query, key = self.W_q(query).unsequeeze(2), self.W_k(key).unsequeeze(1)
        scores = self.W_v(torch.tanh(query + key)).squeeze() # (N, n, m)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf')) # 经过softmax后负无穷的地方会变成0
        attn_weights = F.softmax(scores, dim=-1) # (N, n, m)
        return self.dropout(attn_weights) @ value # (N, n, d_v)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        '''
        :param query: (N, n, d_k)
        :param key: (N, m, d_k)
        :param value: (N, m, d_v)
        :param attn_mask: (N, n, m)
        :return:
        '''
        assert query.size(2) == key.size(2)
        scores = query @ key.transpose(1, 2) / math.sqrt(query.size(2))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        return self.dropout(attn_weights) @ value
    # 由于缩放点积注意力使用较为广泛，因为本文后半部分均采用该评分函数
    # 如果运行过程中出现了nan，可尝试将float('-inf')替换为-1e9这种充分小的负数

# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_size, value_size, dropout=0):
        super().__init__()
        self.attn = ScaleDotProductAttention(dropout)
        self.W_q = nn.Linear(embed_dim, key_size, bias=False)
        self.W_k = nn.Linear(embed_dim, key_size, bias=False)
        self.W_v = nn.Linear(embed_dim, value_size, bias=False)

    def forward(self, X, attn_mask=None):
        '''
        :param X: input sequence, (N, L, embed_dim)
        :param attn_mask: (N, L, L)
        :return:
        '''
        query = self.W_q(X) # (N, L, key_size)
        key = self.W_k(X) # (N, L, key_size)
        value = self.W_v(X) # (N, L, value_size)
        return self.attn(query, key, value, attn_mask) # (N, L, value_size)

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim # 即d_model
        self.num_heads = num_heads # 即注意力头数
        self.head_dim = embed_dim // num_heads # 即每个头的维度
        self.dropout = dropout
        assert self.head_dim * num_heads == embed_dim

        # 初始化W_Q， W_K， W_V， W_O
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 私有化方法来计算缩放点积注意力
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

    # forward 调用私有方法进行前向传播
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

    def _multi_head_attention_forward(self,
                                      query,
                                      key,
                                      value,
                                      dropout_p,
                                      attn_mask=None,
                                      key_padding_mask=None,
                                      training=True):
        # 第一阶段：计算投影后的Q，K，V
        q = self.q_proj(query) # (n, N, embed_dim)
        k = self.k_proj(key) # (m, N, embed_dim)
        v = self.v_proj(value) # (m, N, embed_dim)

        # 第二阶段：attn_mask的维度检查
        n, N, embed_dim = q.size()
        m = key.size(0)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != (n, m):
                    raise RuntimeError
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape != (self.num_heads * N, n, m):
                    raise RuntimeError
            else:
                raise RuntimeError

        # 第三阶段：将attn_mask和key_padding_mask合并
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (N, m)
            key_padding_mask = key_padding_mask.view(N, 1, 1, m).expand(-1, self.num_heads, -1, -1).reshape(self.num_heads * N, 1, m)

            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, -1e9) # 为了防止出现nan，使用充分小的负数

        # 将attn_mask转换成浮点型张量
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, -1e9)
            attn_mask = new_attn_mask

        # 第四阶段：计算注意力
        # 将多头注意力化简为高维单头注意力
        q = q.reshape(n, N * self.num_heads, self.head_dim).transpose(0, 1) # (N * num_heads, n, head_dim)
        k = k.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1) # (N * num_heads, m, head_dim)
        v = v.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1) # (N * num_heads, m, head_dim)

        if not training:
            dropout_p = 0.0

        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = self.out_proj(attn_output)
        attn_output_weights = attn_output_weights.reshape(N, self.num_heads, n, m)
        return attn_output, attn_output_weights

# 两种mask的理解
# key_padding_mask：首先它是一个布尔型张量，其次它只遮盖K，或者说它只遮盖注意力分数QKt(进行softmax前叫分数，softmax后叫权重)
[
    ['a', 'b', 'c', '<pad>', '<pad>'],
    ['x', 'y', '<pad>', '<pad>', '<pad>'],
]

[
    [False, False, False, True, True],
    [False, False, True, True, True],
]

# attn_mask：因为当前时间步不能看到之后时间步的信息，所以需要对当前时间步之后的位置进行mask；可以是布尔型张量也可以是浮点型张量，如果属于前者，
# 则先转化成后者再使用，attn_mask只遮盖QKt的上三角部分。

# 合并两种mask

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
                attn_mask = attn_mask.unsqueeze()
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

    def _scale_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0):
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

# 多头自注意力
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


























