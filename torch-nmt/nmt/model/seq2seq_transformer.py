import math
import random
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

    def forward(self, q, k, v, mask=None):
        bs, ls, hs = q.shape
        # q, k, v: (batch, srclen/tgtlen, hidden)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # q, k, v: (batch, srclen/tgtlen, hidden)
        q = q.view(bs, -1, self.n_heads, self.h_hiddens).permute(0, 2, 1, 3)
        k = k.view(bs, -1, self.n_heads, self.h_hiddens).permute(0, 2, 1, 3)
        v = v.view(bs, -1, self.n_heads, self.h_hiddens).permute(0, 2, 1, 3)
        # q, k, v: (batch, head, srclen/tgtlen, h_hiddens)
        e = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.h_hiddens)
        # e: (batch, head, seqlen, seqlen)
        if mask is not None:
            e = e.masked_fill(mask==0, -1e10)
        a = torch.softmax(e, dim=-1)
        # a: (batch, head, seqlen, seqlen)
        z = torch.matmul(self.dropout(a), v)
        z = z.permute(0, 2, 1, 3).contiguous().view(bs, -1, hs)
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

class PositionalEncoding(nn.Module):
    def __init__(self, n_hiddens, n_position, dropout):
        super().__init__()
        i = torch.arange(n_position).float().reshape(-1, 1)
        j = torch.arange(0, n_hiddens, 2).float()
        self.pe = torch.zeros(1, n_position, n_hiddens)
        self.pe[:,:,0::2] = torch.sin(i / (10000**(j/n_hiddens)))
        self.pe[:,:,1::2] = torch.cos(i / (10000**(j/n_hiddens)))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(n_hiddens)

    def forward(self, x):
        # x: (batch, srclen, embed)
        x = (x * self.scale) + self.pe[:,:x.shape[1],:].to(x.device)
        return self.dropout(x)

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
        self.pos_enc = PositionalEncoding(n_hiddens, n_position, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, n_hiddens, ff_hiddens, dropout)
            for _ in range(n_layers)])

    def forward(self, x, mask):
        bs, ls = x.shape
        # x: (batch, srclen)
        x = self.pos_enc(self.tok_emb(x))
        # x: (batch, srclen, hidden)
        for layer in self.layers:
            x = layer(x, mask)
        # x: (batch, srclen, hidden)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_vocab, n_layers, n_heads, n_hiddens, ff_hiddens,
                 n_position=1000, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, n_hiddens)
        self.pos_enc = PositionalEncoding(n_hiddens, n_position, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, n_hiddens, ff_hiddens, dropout)
            for _ in range(n_layers)])
        self.out = nn.Linear(n_hiddens, n_vocab)

    def forward(self, y, enc_x, mask, enc_mask):
        bs, ls = y.shape
        # y: (batch, srclen)
        y = self.pos_enc(self.tok_emb(y))
        # y: (batch, srclen, embed)
        for layer in self.layers:
            y = layer(y, enc_x, mask, enc_mask)
        o = self.out(y)
        return o

class TransformerSeq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = TransformerEncoder(n_vocab=enc_vocab,
                                          n_layers=kw.get('n_layers', 6),
                                          n_heads=kw.get('n_heads', 8),
                                          n_hiddens=kw.get('n_hiddens', 512),
                                          ff_hiddens=kw.get('ff_hiddens', 2048),
                                          n_position=kw.get('n_position', 100),
                                          dropout=kw.get('dropout', 0.1))
        self.decoder = TransformerDecoder(n_vocab=dec_vocab,
                                          n_layers=kw.get('n_layers', 6),
                                          n_heads=kw.get('n_heads', 8),
                                          n_hiddens=kw.get('n_hiddens', 512),
                                          ff_hiddens=kw.get('ff_hiddens', 1024),
                                          n_position=kw.get('n_position', 100),
                                          dropout=kw.get('dropout', 0.1))

    def make_enc_mask(self, x, x_len):
        bs, ls = x.shape
        m = torch.arange(ls).to(x.device)
        m = m.unsqueeze(0).repeat(bs, 1).lt(x_len.unsqueeze(1))
        # m: (batch, srclen)
        return m.unsqueeze(1).unsqueeze(2)

    def make_dec_mask(self, x, x_len):
        bs, ls = x.shape
        m = torch.arange(ls).to(x.device)
        m = m.unsqueeze(0).repeat(bs, 1).lt(x_len.unsqueeze(1))
        # m: (batch, srclen)
        s = torch.tril(torch.ones((ls, ls)).bool()).to(x.device)
        return m.unsqueeze(1).unsqueeze(2) & s.unsqueeze(0)

    def forward(self, enc_x, enc_len, dec_x, dec_len, teacher_ratio=1.0):
        # enc_x: (batch, seqlen), enc_len: (batch,)
        # dec_x: (batch, seqlen), dec_len: (batch,)
        if teacher_ratio >= 1:
            enc_mask = self.make_enc_mask(enc_x, enc_len)
            dec_mask = self.make_dec_mask(dec_x, dec_len)
            # enc_mask: (batch, 1, 1, srclen)
            # dec_mask: (batch, 1, tgtlen, tgtlen)
            state = self.encoder(enc_x, enc_mask)
            # enc_o: (batch, seqlen, hidden)
            outs = self.decoder(dec_x, state, dec_mask, enc_mask)
        else:
            bs, ls = dec_x.shape
            enc_mask = self.make_enc_mask(enc_x, enc_len)
            # enc_mask: (batch, 1, 1, srclen)
            state = self.encoder(enc_x, enc_mask)
            # enc_o: (batch, seqlen, hidden)
            x_t, pred = [], None
            for t in range(ls):
                if pred is None or (random.random() < teacher_ratio):
                    x_t.append(dec_x[:, t].unsqueeze(1))
                else:
                    x_t.append(pred[:, -1].unsqueeze(1))
                x = torch.cat(x_t, dim=-1).to(dec_x.device)
                x_len = (t+1) * torch.ones(bs).long().to(x.device)
                dec_mask = self.make_dec_mask(x, x_len)
                outs = self.decoder(x, state, dec_mask, enc_mask)
                # outs: (batch, outlen, vocab)
                pred = outs.argmax(2)
                # pred: (batch, outlen)
        return outs

    def predict(self, enc_x, enc_len, dec_x, dec_len, eos_idx=3, maxlen=100):
        bs, dev = dec_x.shape[0], dec_x.device
        enc_mask = self.make_enc_mask(enc_x, enc_len)
        state = self.encoder(enc_x, enc_mask)
        # state: (layers, batch, hiddens)
        x_t, pred = [], None
        pred_len = maxlen * torch.ones(bs).long().to(dev)
        # pred_lens: (batch,)
        for t in range(maxlen):
            if pred is None:
                x_t.append(dec_x[:, t])
            else:
                x_t.append(pred[:, -1])
            x = torch.stack(x_t, dim=-1).to(dev)
            x_len = (t+1) * torch.ones(bs).long().to(dev)
            dec_mask = self.make_dec_mask(x, x_len)
            outs = self.decoder(x, state, dec_mask, enc_mask)
            # outs: (batch, 1, dec_vocab)
            pred = outs.argmax(2)
            # pred: (batch, 1)
            indics = pred_len.gt(t) & pred[:,-1].eq(eos_idx)
            pred_len[indics] = t
        return pred, pred_len


if __name__ == '__main__':
    seq2seq = TransformerSeq2Seq(101, 102, n_layers=2, n_heads=4)
    enc_x, enc_len = torch.randint(101, (8, 10)), torch.randint(1, 10, (8,))
    dec_x, dec_len = torch.randint(102, (8, 11)), torch.randint(1, 11, (8,))
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
