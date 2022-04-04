import torch
import random
from torch import nn
from torch.nn import functional as F


class LuongAttention(nn.Module):
    def __init__(self, n_hiddens, score_fn='dot'):
        super().__init__()
        self.score_fn = score_fn
        if score_fn == 'general':
            self.w = nn.Linear(n_hiddens, n_hiddens)
        elif score_fn == 'concat':
            self.w = nn.Linear(n_hiddens*2, n_hiddens)
            self.v = torch.FloatTensor(n_hiddens)

    def forward(self, q, k, v, mask):
        # q: (batch, 1, hiddens)
        # k: (batch, seqlen, hiddens)
        # v: (batch, seqlen, hiddens)
        if self.score_fn == 'dot':
            a = self.dot(q, k)
        elif self.score_fn == 'general':
            a = self.general(q, k)
        elif self.score_fn == 'concat':
            a = self.concat(q, k)
        # a: (batch, seqlen)
        w = F.softmax(a, dim=-1)
        c = torch.bmm(w.unsqueeze(1), v)
        # c: (batch, 1, hiddens)
        return c

    def dot(self, q, k):
        return torch.sum(q*k, dim=-1)

    def general(self, q, k):
        e = self.w(k)
        return torch.sum(q*e, dim=-1)

    def concat(self, q, k):
        q = q.repeat(1, k.shape[1], 1)
        e = torch.cat([q, k], dim=-1)
        w = torch.tanh(self.w(e))
        return torch.sum(self.v * w, dim=-1)

class EncoderLuong(nn.Module):
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
        self.birnn = use_birnn

    def forward(self, x):
        # x: (batch, seqlen)
        x = self.dropout(self.emb(x))
        # x: (batch, seqlen, embed)
        outs, hidden = self.rnn(x)
        # outs: (batch, seqlen, hiddens*dir)
        # hidden: (layers*dir, batch, hiddens)
        if self.birnn:
            # hidden: (layers*2, batch, hiddens)
            _, batch_size, n_hiddens = hidden.shape
            hidden = hidden.view(-1, 2, batch_size, n_hiddens)
            # hidden: (layers, 2, batch, hiddens)
            hidden = torch.stack([torch.cat((h[0], h[1]), dim=1)
                                  for h in hidden])
        # hidden: (layers, batch, hiddens*dir)
        return (outs, hidden)

class DecoderLuong(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 use_birnn=False):
        super().__init__()
        if use_birnn: n_hiddens *= 2
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.attn = LuongAttention(n_hiddens)
        self.dense = nn.Linear(n_embed+n_hiddens*2, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state, mask):
        # x: (batch, seqlen)
        enc_outs, hidden = state
        # enc_outs: (batch, seqlen, hiddens)
        # hidden: (layers, batch, hiddens)
        e = self.dropout(self.emb(x))
        # e: (batch, seqlen, embed)
        o, h = self.rnn(e, hidden)
        # o: (batch, seqlen, hiddens)
        # h: (layers, batch, hiddens)
        c = self.attn(o, enc_outs, enc_outs, mask)
        # c: (batch, 1, hiddens)
        out = self.dense(torch.cat([e, c, o], dim=-1))
        return out, (enc_outs, hidden)

class Seq2SeqLuong(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = EncoderLuong(n_vocab=enc_vocab,
                                       n_embed=kw.get('enc_embed', 256),
                                       n_hiddens=kw.get('n_hiddens', 512),
                                       n_layers=kw.get('n_layers', 1),
                                       use_birnn=kw.get('use_birnn', False),
                                       dropout=kw.get('dropout', 0.0))
        self.decoder = DecoderLuong(n_vocab=dec_vocab,
                                       n_embed=kw.get('dec_embed', 256),
                                       n_hiddens=kw.get('n_hiddens', 512),
                                       use_birnn=kw.get('use_birnn', False),
                                       n_layers=kw.get('n_layers', 1),
                                       dropout=kw.get('dropout', 0.0))
        self.enc_pad = kw.get('enc_pad', 1)

    def make_enc_mask(self, enc_x):
        # enc_x: (batch, seqlen)
        enc_mask = (enc_x != self.enc_pad).unsqueeze(1)
        # enc_mask: (batch, 1, seqlen)
        return enc_mask

    def forward(self, enc_x, dec_x, teacher_ratio=0.5):
        # enc_x: (batch, seqlen)
        # dec_x: (batch, seqlen)
        mask = self.make_enc_mask(enc_x)
        state = self.encoder(enc_x)
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
    seq2seq = Seq2SeqLuong(101, 102, n_layers=2, use_birnn=True)
    enc_x = torch.randint(101, (32, 10))
    dec_x = torch.randint(102, (32, 11))
    outs = seq2seq(enc_x, dec_x)
    print(outs.shape)
