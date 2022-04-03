import torch
import random
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, n_hiddens, n_dir=1):
        super().__init__()
        self.w = nn.Linear(n_hiddens*(n_dir+1), n_hiddens)
        self.v = nn.Linear(n_hiddens, 1, bias=False)

    def forward(self, q, k, v):
        # q: (batch, hiddens)
        # k: (batch, seqlen, hiddens*dir)
        # v: (batch, seqlen, hiddens*dir)
        q = q.unsqueeze(1).repeat(1, k.shape[1], 1)
        # q: (batch, seqlen, hiddens)
        a = self.w(torch.cat([q, k], dim=-1))
        # a: (batch, seqlen, hiddens)
        w = self.v(torch.tanh(a)).permute(0, 2, 1)
        # w: (batch, 1, seqlen)
        out = torch.bmm(F.softmax(w, dim=-1), v)
        # out: (batch, 1, hiddens)
        return out

class EncoderBahdanau(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 enc_birnn=False):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          bidirectional=enc_birnn,
                          dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.birnn = enc_birnn
        if self.birnn :
            self.dense = nn.Linear(n_layers*2, n_layers)

    def forward(self, x):
        # x: (batch, seqlen)
        x = self.dropout(self.emb(x))
        # x: (batch, seqlen, embed)
        outs, hidden = self.rnn(x)
        # outs: (batch, seqlen, hiddens*dir)
        # hidden: (layers*dir, batch, hiddens)
        if self.birnn:
            # hidden: (layers*2, batch, hiddens)
            hidden = self.dense(hidden.permute(1, 2, 0))
            # hidden: (batch, hiddens, layers)
            hidden = torch.tanh(hidden.permute(2, 0, 1))
            # hidden: (layers, batch, hiddens)
        return (outs, hidden)

class DecoderBahdanau(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 enc_birnn=False):
        super().__init__()
        n_dir = 2 if enc_birnn else 1
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed+n_hiddens*n_dir, n_hiddens,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.attn = BahdanauAttention(n_hiddens, n_dir)
        self.dense = nn.Linear(n_embed+n_hiddens*2, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state):
        # x: (batch, seqlen)
        enc_outs, hidden = state
        # enc_outs: (batch, seqlen, hiddens*dir)
        # hidden: (layers, batch, hiddens)
        e = self.dropout(self.emb(x))
        # e: (batch, seqlen, embed)
        c = self.attn(hidden[-1], enc_outs, enc_outs)
        # c: (batch, seqlen, hiddens*dir)
        a = torch.cat([e, c], dim=-1)
        # a: (batch, seqlen, embed+hiddens*dir)
        out, hidden = self.rnn(a, hidden)
        # out: (batch, seqlen, hiddens)
        # hidden: (layers, batch, hiddens)
        h = hidden[-1].unsqueeze(1).repeat(1, e.shape[1], 1)
        # h: (batch, seqlen, hiddens)
        out = torch.cat([e, h, out], dim=-1)
        out = self.dense(out)
        # out: (batch, seqlen, vocab)
        return out, (enc_outs, hidden)

class Seq2SeqBahdanau(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = EncoderBahdanau(n_vocab=enc_vocab,
                                       n_embed=kw.get('enc_embed', 256),
                                       n_hiddens=kw.get('enc_hiddens', 512),
                                       n_layers=kw.get('n_layers', 1),
                                       enc_birnn=kw.get('enc_birnn', False),
                                       dropout=kw.get('dropout', 0.0))
        self.decoder = DecoderBahdanau(n_vocab=dec_vocab,
                                       n_embed=kw.get('dec_embed', 256),
                                       n_hiddens=kw.get('dec_hiddens', 512),
                                       enc_birnn=kw.get('enc_birnn', False),
                                       n_layers=kw.get('n_layers', 1),
                                       dropout=kw.get('dropout', 0.0))

    def forward(self, enc_x, dec_x, teacher_ratio=0.5):
        # enc_x: (batch, seqlen)
        # dec_x: (batch, seqlen)
        seqlen = dec_x.shape[1]
        state = self.encoder(enc_x)
        pred, outs = None, []
        for t in range(seqlen):
            if pred is None or (random.random() < teacher_ratio):
                x = dec_x[:, t]
            else:
                x = pred
            # x: (batch,)
            out, state = self.decoder(x.unsqueeze(-1), state)
            # out: (batch, 1, vocab)
            pred = out.argmax(2).squeeze(1)
            # pred: (batch,)
            outs.append(out)
        return torch.cat(outs, dim=1)

if __name__ == '__main__':
    seq2seq = Seq2SeqBahdanau(101, 102, n_layers=2, enc_birnn=True)
    enc_x = torch.randint(101, (32, 10))
    dec_x = torch.randint(102, (32, 11))
    outs = seq2seq(enc_x, dec_x)
    print(outs.shape)
