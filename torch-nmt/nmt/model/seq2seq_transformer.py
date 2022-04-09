import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_hidden, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.h_hidden = n_hidden // n_head
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.w_o = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor([self.h_hidden]))

    def forward(self, q, k, v, mask=None):
        bs, ls, hs = q.shape
        # q, k, v: (batch, srclen/tgtlen, hidden)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # q, k, v: (batch, srclen/tgtlen, hidden)
        q = q.view(bs, -1, self.n_head, self.h_hidden).transpose(1, 2)
        k = k.view(bs, -1, self.n_head, self.h_hidden).transpose(1, 2)
        v = v.view(bs, -1, self.n_head, self.h_hidden).transpose(1, 2)
        # q, k, v: (batch, head, srclen/tgtlen, h_hidden)
        e = torch.matmul(q, k.transpose(2, 3)) / self.scale
        # e: (batch, head, seqlen, seqlen)
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)
        a = torch.softmax(e, dim=-1)
        # a: (batch, head, seqlen, seqlen)
        z = torch.matmul(self.dropout(a), v)
        z = z.transpose(1, 2).reshape(bs, -1, hs)
        # z: (batch, seqlen, hidden)
        o = self.w_o(z)
        # o: (batch, seqlen, hidden)
        return o, a

class PositionwiseFFN(nn.Module):
    def __init__(self, n_hidden, f_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(n_hidden, f_hidden)
        self.w_2 = nn.Linear(f_hidden, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seqlen, hidden)
        x = self.relu(self.w_1(x))
        # x: (batch, seqlen, f_hidden)
        x = self.w_2(self.dropout(x))
        # x: (batch, seqlen, hidden)
        return x

class AddNorm(nn.Module):
    def __init__(self, n_shape, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))

class EncoderLayer(nn.Module):
    def __init__(self, n_head, n_hidden, ff_hidden, dropout=0.1):
        super().__init__()
        self.att = MultiHeadAttention(n_hidden, n_head, dropout)
        self.att_norm = AddNorm(n_hidden, dropout)
        self.ffn = PositionwiseFFN(n_hidden, ff_hidden, dropout)
        self.ffn_norm = AddNorm(n_hidden, dropout)

    def forward(self, x, mask):
        x_, _ = self.att(x, x, x, mask)
        x = self.att_norm(x, x_)
        x_ = self.ffn(x)
        x = self.ffn_norm(x, x_)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_head, n_hidden, ff_hidden, dropout=0.1):
        super().__init__()
        self.att = MultiHeadAttention(n_hidden, n_head, dropout)
        self.att_norm = AddNorm(n_hidden, dropout)
        self.attE = MultiHeadAttention(n_hidden, n_head, dropout)
        self.attE_norm = AddNorm(n_hidden, dropout)
        self.ffn = PositionwiseFFN(n_hidden, ff_hidden, dropout)
        self.ffn_norm = AddNorm(n_hidden, dropout)

    def forward(self, x, enc_x, mask, enc_mask):
        x_, _ = self.att(x, x, x, mask)
        x = self.att_norm(x, x_)
        x_, _ = self.attE(x, enc_x, enc_x, enc_mask)
        x =  self.attE_norm(x, x_)
        x_ = self.ffn(x)
        x = self.ffn_norm(x, x_)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_vocab, n_layers, n_head, n_hidden, ff_hidden,
                 n_pos=100, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, n_hidden)
        self.pos_emb = nn.Embedding(n_pos, n_hidden)
        self.layers = nn.ModuleList([
            EncoderLayer(n_head, n_hidden, ff_hidden, dropout)
            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        bs, ls = x.shape
        # x: (batch, srclen)
        p = torch.arange(ls).unsqueeze(0).repeat(bs, 1)
        x = self.dropout(self.pos_emb(p) + self.tok_emb(x))
        # x: (batch, srclen, hidden)
        for layer in self.layers:
            x = layer(x, mask)
        # x: (batch, srclen, hidden)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_vocab, n_layers, n_head, n_hidden, ff_hidden,
                 n_pos=100, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, n_hidden)
        self.pos_emb = nn.Embedding(n_pos, n_hidden)
        self.layers = nn.ModuleList([
            DecoderLayer(n_head, n_hidden, ff_hidden, dropout)
            for _ in range(n_layers)])
        self.dense = nn.Linear(n_hidden, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_x, mask, enc_mask):
        bs, ls = x.shape
        # x: (batch, srclen)
        p = torch.arange(ls).unsqueeze(0).repeat(bs, 1)
        x = self.dropout(self.pos_emb(p) + self.tok_emb(x))
        # x: (batch, srclen, embed)
        for layer in self.layers:
            x = layer(x, enc_x, mask, enc_mask)
        o = self.dense(x)
        return o

class TransformerSeq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = TransformerEncoder(n_vocab=enc_vocab,
                                          n_layers=kw.get('n_layers', 2),
                                          n_head=kw.get('n_head', 4),
                                          n_hidden=kw.get('n_hidden', 512),
                                          ff_hidden=kw.get('ff_hidden', 1024),
                                          n_pos=kw.get('n_pos', 100),
                                          dropout=kw.get('dropout', 0.0))
        self.decoder = TransformerDecoder(n_vocab=dec_vocab,
                                          n_layers=kw.get('n_layers', 2),
                                          n_head=kw.get('n_head', 4),
                                          n_hidden=kw.get('n_hidden', 512),
                                          ff_hidden=kw.get('ff_hidden', 1024),
                                          n_pos=kw.get('n_pos', 100),
                                          dropout=kw.get('dropout', 0.0))

    def make_enc_mask(self, x, x_len):
        bs, ls = x.shape
        m = torch.arange(ls).unsqueeze(0).repeat(bs, 1).lt(x_len.unsqueeze(1))
        # m: (batch, srclen)
        return m.unsqueeze(1).unsqueeze(2)

    def make_dec_mask(self, x, x_len):
        bs, ls = x.shape
        m = torch.arange(ls).unsqueeze(0).repeat(bs, 1).lt(x_len.unsqueeze(1))
        # m: (batch, srclen)
        s = torch.tril(torch.ones((ls, ls))).bool()
        return m.unsqueeze(1).unsqueeze(2) & s.unsqueeze(0)

    def forward(self, enc_x, enc_len, dec_x, dec_len, teacher_ratio=0.0):
        enc_mask = self.make_enc_mask(enc_x, enc_len)
        dec_mask = self.make_dec_mask(dec_x, dec_len)
        # enc_mask: (batch, 1, 1, srclen)
        # dec_mask: (batch, 1, tgtlen, tgtlen)
        enc_o = self.encoder(enc_x, enc_mask)
        # enc_o: (batch, seqlen, hidden)
        out = self.decoder(dec_x, enc_o, dec_mask, enc_mask)
        return out

if __name__ == '__main__':
    seq2seq = TransformerSeq2Seq(101, 102, n_layers=2, n_head=4)
    enc_x, enc_len = torch.randint(101, (8, 10)), torch.randint(1, 10, (8,))
    dec_x, dec_len = torch.randint(102, (8, 11)), torch.randint(1, 11, (8,))
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
