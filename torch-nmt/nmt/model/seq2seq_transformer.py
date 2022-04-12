import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_hiddens, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.h_hiddens = n_hiddens // n_heads
        self.w_q = nn.Linear(n_hiddens, n_hiddens)
        self.w_k = nn.Linear(n_hiddens, n_hiddens)
        self.w_v = nn.Linear(n_hiddens, n_hiddens)
        self.w_o = nn.Linear(n_hiddens, n_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.h_hiddens]))

    def forward(self, q, k, v, mask=None):
        bs, ls, hs = q.shape
        # q, k, v: (batch, srclen/tgtlen, hidden)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # q, k, v: (batch, srclen/tgtlen, hidden)
        q = q.view(bs, -1, self.n_heads, self.h_hiddens).transpose(1, 2)
        k = k.view(bs, -1, self.n_heads, self.h_hiddens).transpose(1, 2)
        v = v.view(bs, -1, self.n_heads, self.h_hiddens).transpose(1, 2)
        # q, k, v: (batch, head, srclen/tgtlen, h_hiddens)
        e = torch.matmul(q, k.transpose(2, 3)) / self.scale
        # e: (batch, head, seqlen, seqlen)
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)
        a = torch.softmax(e, dim=-1)
        # a: (batch, head, seqlen, seqlen)
        z = torch.matmul(self.dropout(a), v)
        z = z.transpose(1, 2).contiguous().view(bs, -1, hs)
        # z: (batch, seqlen, hidden)
        o = self.w_o(z)
        # o: (batch, seqlen, hidden)
        return o, a

class PositionwiseFFN(nn.Module):
    def __init__(self, n_hiddens, f_hiddens, dropout):
        super().__init__()
        self.w_1 = nn.Linear(n_hiddens, f_hiddens)
        self.w_2 = nn.Linear(f_hiddens, n_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seqlen, hidden)
        x = self.relu(self.w_1(x))
        # x: (batch, seqlen, f_hiddens)
        x = self.w_2(self.dropout(x))
        # x: (batch, seqlen, hidden)
        return x

class AddNorm(nn.Module):
    def __init__(self, n_shape, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(n_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))

class PositionEncoding(nn.Module):
    def __init__(self, n_position, n_hiddens):
        super().__init__()
        self.pos_emb = nn.Embedding(n_position, n_hiddens)
        self.scale = torch.sqrt(torch.FloatTensor([n_hiddens]))

    def forward(self, x):
        bs, ls, _ = x.shape
        # x: (batch, srclen, embed)
        p = torch.arange(ls).unsqueeze(0).repeat(bs, 1)
        # p: (batch, seqlen)
        return (x * self.scale) + self.pos_emb(p)

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, n_hiddens, ff_hiddens, dropout):
        super().__init__()
        self.att = MultiHeadAttention(n_hiddens, n_heads, dropout)
        self.att_norm = AddNorm(n_hiddens, dropout)
        self.ffn = PositionwiseFFN(n_hiddens, ff_hiddens, dropout)
        self.ffn_norm = AddNorm(n_hiddens, dropout)

    def forward(self, x, mask):
        x_, _ = self.att(x, x, x, mask)
        x = self.att_norm(x, x_)
        x_ = self.ffn(x)
        x = self.ffn_norm(x, x_)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, n_hiddens, ff_hiddens, dropout):
        super().__init__()
        self.att = MultiHeadAttention(n_hiddens, n_heads, dropout)
        self.att_norm = AddNorm(n_hiddens, dropout)
        self.attE = MultiHeadAttention(n_hiddens, n_heads, dropout)
        self.attE_norm = AddNorm(n_hiddens, dropout)
        self.ffn = PositionwiseFFN(n_hiddens, ff_hiddens, dropout)
        self.ffn_norm = AddNorm(n_hiddens, dropout)

    def forward(self, y, enc_x, mask, enc_mask):
        y_, _ = self.att(y, y, y, mask)
        y = self.att_norm(y, y_)
        y_, _ = self.attE(y, enc_x, enc_x, enc_mask)
        y =  self.attE_norm(y, y_)
        y_ = self.ffn(y)
        y = self.ffn_norm(y, y_)
        return y

class TransformerEncoder(nn.Module):
    def __init__(self, n_vocab, n_layers, n_heads, n_hiddens, ff_hiddens,
                 n_position=100, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, n_hiddens)
        self.pos_enc = PositionEncoding(n_position, n_hiddens)
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, n_hiddens, ff_hiddens, dropout)
            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        bs, ls = x.shape
        # x: (batch, srclen)
        x = self.dropout(self.pos_enc(self.tok_emb(x)))
        # x: (batch, srclen, hidden)
        for layer in self.layers:
            x = layer(x, mask)
        # x: (batch, srclen, hidden)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_vocab, n_layers, n_heads, n_hiddens, ff_hiddens,
                 n_position=100, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, n_hiddens)
        self.pos_enc = PositionEncoding(n_position, n_hiddens)
        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, n_hiddens, ff_hiddens, dropout)
            for _ in range(n_layers)])
        self.out = nn.Linear(n_hiddens, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, enc_x, mask, enc_mask):
        bs, ls = y.shape
        # y: (batch, srclen)
        y = self.dropout(self.pos_enc(self.tok_emb(y)))
        # y: (batch, srclen, embed)
        for layer in self.layers:
            y = layer(y, enc_x, mask, enc_mask)
        o = self.out(y)
        return o

class TransformerSeq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = TransformerEncoder(n_vocab=enc_vocab,
                                          n_layers=kw.get('n_layers', 2),
                                          n_heads=kw.get('n_heads', 4),
                                          n_hiddens=kw.get('n_hiddens', 512),
                                          ff_hiddens=kw.get('ff_hiddens', 1024),
                                          n_position=kw.get('n_position', 100),
                                          dropout=kw.get('dropout', 0.0))
        self.decoder = TransformerDecoder(n_vocab=dec_vocab,
                                          n_layers=kw.get('n_layers', 2),
                                          n_heads=kw.get('n_heads', 4),
                                          n_hiddens=kw.get('n_hiddens', 512),
                                          ff_hiddens=kw.get('ff_hiddens', 1024),
                                          n_position=kw.get('n_position', 100),
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
    seq2seq = TransformerSeq2Seq(101, 102, n_layers=2, n_heads=4)
    enc_x, enc_len = torch.randint(101, (8, 10)), torch.randint(1, 10, (8,))
    dec_x, dec_len = torch.randint(102, (8, 11)), torch.randint(1, 11, (8,))
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
