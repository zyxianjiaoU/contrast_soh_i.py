import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


# class PositionalEncoding(nn.Module):
#     r"""Inject some information about the relative or absolute position of the tokens in the sequence.
#         The positional encodings have the same dimension as the embeddings, so that the two can be summed.
#         Here, we use sine and cosine functions of different frequencies.
#         math:
#         \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#         \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#         \text{where pos is the word position and i is the embed idx)
#     Args:
#         d_model: the embed dim (required).
#         dropout: the dropout value (default=0.1).
#         max_len: the max. length of the incoming sequence (default=5000).
#     """
#     def __init__(self, dropout):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#     def forward(self, x, position_id):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [batch size, sequence length, embed dim]
#             output: [batch size, sequence length, embed dim]
#         """
#         bat_len, d_model = x.size()[1], x.size()[2]
#         pe = torch.zeros(bat_len, d_model)
#         # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
#         pe = position_id.unsqueeze(-1) * div_term
#         pe[:, :, 0::2] = torch.sin(pe[:, :, 0::2])
#         pe[:, :, 1::2] = torch.cos(pe[:, :, 1::2])
#         self.register_buffer('pe', pe)
#         x = x + Variable(pe, requires_grad=False)
#         return self.dropout(x)
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
        math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, max_len, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # self.pe = self.pe.unsqueeze(0)
        x = x + Variable(self.pe[:, :, :x.size()[2]], requires_grad=False)
        return self.dropout(x)


# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embeddings, self).__init__()
#         # vocab代表词表大小, d_model指词嵌入的维度
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model
#
#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)


def subsequent_mask(size):
    mask = torch.empty(size, size)  # 2 = channel of soh
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask


def get_pad_mask(seq_k, pad_idx):
    pad = torch.sum(seq_k, dim=2)
    mask = torch.eq(pad, pad_idx * seq_k.size()[2]).int()
    return mask.unsqueeze(1)


def create_mask(targets, max_len):
    sub_mask = subsequent_mask(max_len)
    padding_mask = get_pad_mask(targets, -20)
    mask = torch.max(padding_mask, sub_mask)
    return mask.int()



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)    # 获取词嵌入维度d_k
    # key需要将最后两个维度转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask = mask.unsqueeze(1)
        try:
            scores = scores.masked_fill(mask == float("-inf"), -1e10)
        except:
            print('kkk')
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:     # 随机置0
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiheadAttention(nn.Module):
    def __init__(self, head, embedding_dim, drop_out=0.1):
        super(MultiheadAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding = embedding_dim
        self.linears = clone(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(feature))
        self.b2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class ResConnect(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(ResConnect, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, res_layer):
        res  = self.dropout(res_layer(self.norm(x)))
        x = x.view(res.size(0), -1, res.size(2))
        return  res + x


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = self_attn
        self.feedforward = feed_forward
        self.sub_layer = clone(ResConnect(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sub_layer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sub_layer[1](x, self.feedforward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(ResConnect(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        # return self.project(x)
        x = x[:, -1].squeeze(1)
        return F.softmax(self.project(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, position, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.position = position

    def forward(self, source, target, source_mask, target_mask):
        return self.generator(self.decode(self.encode(source, source_mask), source_mask, target, target_mask))

    def encode(self, source, source_mask):
        return self.encoder(self.position(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.position(target), memory, source_mask, target_mask)


def make_model_tf(target_vocab, d_model, head, N=6, d_ff=1024, dropout=0.1):
    c = copy.deepcopy
    attn = MultiheadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(max_len=5000, d_model=d_model, dropout=dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        c(position),
        Generator(d_model, target_vocab),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class DecoderLayerGpt(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayerGpt, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(ResConnect(size, dropout), 2)

    def forward(self, x, target_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderGpt(nn.Module):
    def __init__(self, layer, N):
        super(DecoderGpt, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, target_mask):
        for layer in self.layers:
            x = layer(x, target_mask)
        return self.norm(x)

class Gpt(nn.Module):
    def __init__(self,  decoder, position, generator):
        super(Gpt, self).__init__()
        self.decoder = decoder
        self.generator = generator
        self.position = position

    def forward(self, target, target_mask, cycle):
        return self.generator(self.decoder(self.position(target, cycle), target_mask))


def make_gpt(target_vocab, d_model, head, N=6, d_ff=1024, dropout=0.1):
    c = copy.deepcopy
    attn = MultiheadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(5000, d_model, dropout)
    model = Gpt(
        DecoderGpt(DecoderLayerGpt(d_model, c(attn), c(ff), dropout), N),
        c(position),
        Generator(d_model, target_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def get_para_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

