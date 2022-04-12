import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class BahdanauAttention(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.w_a = nn.Linear(n_hiddens*2, n_hiddens)
        self.v_a = nn.Linear(n_hiddens, 1, bias=False)

    def forward(self, q, k, v, mask=None):
        # q: (batch, hiddens0)
        # k: (batch, seqlen, hiddens)
        # v: (batch, seqlen, hiddens)
        n_dir = k.shape[2] // q.shape[1]
        q = q.unsqueeze(1).repeat(1, k.shape[1], n_dir)
        # q: (batch, seqlen, hiddens)
        cat = torch.cat([q, k], dim=-1)
        # cat: (batch, seqlen, hiddens)
        e = self.v_a(torch.tanh(self.w_a(cat)))
        # e: (batch, seqlen, 1)
        e = e.permute(0, 2, 1)
        # e: (batch, 1, seqlen)
        if mask is not None:
            e = e.masked_fill(mask==0, -1e10)
        a = F.softmax(e, dim=-1)
        # a: (batch, 1, seqlen)
        c = torch.bmm(a, v)
        # c: (batch, 1, hiddens)
        return c

class BahdanauEncoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 use_birnn=False):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          bidirectional=use_birnn,
                          batch_first=True,
                          dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_birnn = use_birnn
        if self.use_birnn :
            self.dense = nn.Linear(n_layers*2, n_layers)

    def forward(self, x, l):
        bs, ls = x.shape
        # x: (batch, seqlen), l: (batch,)
        e = self.dropout(self.emb(x))
        # e: (batch, seqlen, embed)
        e = pack_padded_sequence(e, l.cpu(), batch_first=True,
                                 enforce_sorted=False)
        o, h = self.rnn(e)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=ls)
        # o: (batch, seqlen, hiddens*dir)
        # h: (layers*dir, batch, hiddens)
        if self.use_birnn:
            # h: (layers*2, batch, hiddens)
            h = self.dense(h.permute(1, 2, 0))
            # h: (batch, hiddens, layers)
            h = torch.tanh(h.permute(2, 0, 1))
        # o: (batch, seqlen, hiddens*dir)
        # h: (layers, batch, hiddens)
        return (o, h)

class BahdanauDecoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 use_birnn=False):
        super().__init__()
        self.n_dir = 2 if use_birnn else 1
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed+n_hiddens*self.n_dir, n_hiddens,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)
        self.att = BahdanauAttention(n_hiddens*self.n_dir)
        self.out = nn.Linear(n_hiddens, n_vocab)
        # self.out = nn.Linear(n_embed+n_hiddens*2, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, state, mask):
        # y: (batch, seqlen)
        ctx, s = state
        # ctx: (batch, seqlen, hiddens*dir)
        # s: (layers, batch, hiddens)
        e = self.dropout(self.emb(y))
        # e: (batch, seqlen, embed)
        c = self.att(s[-1], ctx, ctx, mask)
        # c: (batch, seqlen, hiddens)
        cat = torch.cat([e, c], dim=-1)
        # cat: (batch, seqlen, embed+hiddens*dir)
        o, s = self.rnn(cat, s)
        # o: (batch, seqlen, hiddens)
        # s: (layers, batch, hiddens)
        ## h = h[-1].unsqueeze(1).repeat(1, e.shape[1], 1)
        ## o = torch.cat([e, h, o], dim=-1)
        # o: (batch, seqlen, embed+hiddens*2)
        o = self.out(o)
        # o: (batch, seqlen, vocab)
        # ctx: (batch, seqlen, hiddens*dir)
        # s: (layers, batch, hiddens)
        return o, (ctx, s)

class BahdanauSeq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = BahdanauEncoder(n_vocab=enc_vocab,
                                       n_embed=kw.get('enc_embed', 256),
                                       n_hiddens=kw.get('n_hiddens', 512),
                                       n_layers=kw.get('n_layers', 1),
                                       use_birnn=kw.get('use_birnn', False),
                                       dropout=kw.get('dropout', 0.0))
        self.decoder = BahdanauDecoder(n_vocab=dec_vocab,
                                       n_embed=kw.get('dec_embed', 256),
                                       n_hiddens=kw.get('n_hiddens', 512),
                                       n_layers=kw.get('n_layers', 1),
                                       use_birnn=kw.get('use_birnn', False),
                                       dropout=kw.get('dropout', 0.0))

    def make_enc_mask(self, x, x_len):
        # x: (batch, seqlen)
        bs, ls = x.shape
        m = torch.arange(ls).to(x.device)
        m = m.unsqueeze(0).repeat(bs, 1).lt(x_len.unsqueeze(1))
        # m: (batch, 1, seqlen)
        return m.unsqueeze(1)

    def forward(self, enc_x, enc_len, dec_x, dec_len, teacher_ratio=0.5):
        # enc_x: (batch, seqlen), enc_len: (batch,)
        # dec_x: (batch, seqlen), dec_len: (batch,)
        enc_mask = self.make_enc_mask(enc_x, enc_len)
        state = self.encoder(enc_x, enc_len)
        pred, outs = None, []
        for t in range(dec_x.shape[1]):
            if pred is None or (random.random() < teacher_ratio):
                x = dec_x[:, t]
            else:
                x = pred
            # x: (batch,)
            out, state = self.decoder(x.unsqueeze(-1), state, enc_mask)
            outs.append(out)
            # out: (batch, 1, vocab)
            pred = out.argmax(2).squeeze(1)
            # pred: (batch,)
        return torch.cat(outs, dim=1)

if __name__ == '__main__':
    seq2seq = BahdanauSeq2Seq(101, 102, n_layers=2, use_birnn=True)
    enc_x, enc_len = torch.randint(101, (32, 10)), torch.randint(1, 10, (32,))
    dec_x, dec_len = torch.randint(102, (32, 11)), torch.randint(1, 10, (32,))
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
