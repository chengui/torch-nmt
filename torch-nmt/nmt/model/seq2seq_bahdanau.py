import torch
import random
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.w = nn.Linear(n_hiddens*2, n_hiddens)
        self.v = nn.Linear(n_hiddens, 1, bias=False)

    def forward(self, q, k, v, mask):
        # q: (batch, seqlen, hiddens)
        # k: (batch, seqlen, hiddens)
        # v: (batch, seqlen, hiddens)
        a = self.w(torch.cat([q, k], dim=-1))
        # a: (batch, seqlen, hiddens)
        w = self.v(torch.tanh(a)).permute(0, 2, 1)
        # w: (batch, 1, seqlen)
        w = F.softmax(w.masked_fill(mask==0, -1e10), dim=-1)
        # w: (batch, 1, seqlen)
        out = torch.bmm(w, v)
        # out: (batch, 1, hiddens)
        return out

class BahdanauEncoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 use_birnn=False):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          bidirectional=use_birnn,
                          dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_birnn = use_birnn
        if self.use_birnn :
            self.dense = nn.Linear(n_layers*2, n_layers)

    def forward(self, x, l):
        # x: (batch, seqlen), l: (batch,)
        e = self.dropout(self.emb(x))
        # e: (batch, seqlen, embed)
        o, h = self.rnn(e)
        # o: (batch, seqlen, hiddens*dir)
        # h: (layers*dir, batch, hiddens)
        if self.use_birnn:
            # h: (layers*2, batch, hiddens)
            h = self.dense(h.permute(1, 2, 0))
            # h: (batch, hiddens, layers)
            h = torch.tanh(h.permute(2, 0, 1))
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
                          dropout=dropout,
                          batch_first=True)
        self.attn = BahdanauAttention(n_hiddens*self.n_dir)
        self.dense = nn.Linear(n_embed+n_hiddens*2, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state, mask):
        # x: (batch, seqlen)
        enc_outs, hidden = state
        # enc_outs: (batch, seqlen, hiddens*dir)
        # hidden: (layers, batch, hiddens)
        e = self.dropout(self.emb(x))
        # e: (batch, seqlen, embed)
        h = hidden[-1].unsqueeze(1).repeat(1, enc_outs.shape[1], self.n_dir)
        c = self.attn(h, enc_outs, enc_outs, mask)
        # c: (batch, seqlen, hiddens*dir)
        a = torch.cat([e, c], dim=-1)
        # a: (batch, seqlen, embed+hiddens*dir)
        o, hidden = self.rnn(a, hidden)
        # o: (batch, seqlen, hiddens)
        # hidden: (layers, batch, hiddens)
        h = hidden[-1].unsqueeze(1).repeat(1, e.shape[1], 1)
        # h: (batch, seqlen, hiddens)
        o = torch.cat([e, h, o], dim=-1)
        out = self.dense(o)
        # out: (batch, seqlen, vocab)
        return out, (enc_outs, hidden)

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
                                       use_birnn=kw.get('use_birnn', False),
                                       n_layers=kw.get('n_layers', 1),
                                       dropout=kw.get('dropout', 0.0))

    def make_enc_mask(self, enc_len, maxlen):
        # enc_len: (batch,)
        m = torch.arange(maxlen).unsqueeze(0).repeat(enc_len.shape[0], 1)
        enc_mask = m.lt(enc_len.unsqueeze(1)).unsqueeze(1)
        # enc_mask: (batch, 1, seqlen)
        return enc_mask

    def forward(self, enc_x, enc_len, dec_x, dec_len, teacher_ratio=0.5):
        # enc_x: (batch, seqlen), enc_len: (batch,)
        # dec_x: (batch, seqlen), dec_len: (batch,)
        mask = self.make_enc_mask(enc_len, enc_x.shape[1])
        state = self.encoder(enc_x, enc_len)
        pred, outs = None, []
        for t in range(dec_x.shape[1]):
            if pred is None or (random.random() < teacher_ratio):
                x = dec_x[:, t]
            else:
                x = pred
            # x: (batch,)
            out, state = self.decoder(x.unsqueeze(-1), state, mask)
            # out: (batch, 1, vocab)
            pred = out.argmax(2).squeeze(1)
            # pred: (batch,)
            outs.append(out)
        return torch.cat(outs, dim=1)

if __name__ == '__main__':
    seq2seq = BahdanauSeq2Seq(101, 102, n_layers=2, use_birnn=True)
    enc_x, enc_len = torch.randint(101, (8, 10)), torch.randint(1, 10, (8,))
    dec_x, dec_len = torch.randint(102, (8, 11)), torch.randint(1, 10, (8,))
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
