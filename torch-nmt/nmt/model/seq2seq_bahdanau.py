import torch
import random
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Linear(hid_dim*2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, q, k, v):
        # q: (batch, hid_dim)
        # k: (batch, seqlen, hid_dim)
        # v: (batch, seqlen, hid_dim)
        q = q.unsqueeze(1).repeat(1, k.shape[1], 1)
        # q: (batch, seqlen, hid_dim)
        a = self.w(torch.cat([q, k], dim=-1))
        # a: (batch, seqlen, hid_dim)
        w = self.v(torch.tanh(a)).permute(0, 2, 1)
        # w: (batch, 1, seqlen)
        out = torch.bmm(F.softmax(w, dim=-1), v)
        # out: (batch, 1, hid_dim)
        return out

class EncoderBahdanau(nn.Module):
    def __init__(self, in_dim, emb_dim, hid_dim, n_layers, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(in_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seqlen)
        x = self.dropout(self.emb(x))
        # x: (batch, seqlen, emb_dim)
        outs, hidden = self.rnn(x)
        # outs: (batch, seqlen, hid_dim)
        # hidden: (layers, batch, hid_dim)
        return (outs, hidden)

class DecoderBahdanau(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(out_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim+hid_dim, hid_dim,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.attn = BahdanauAttention(hid_dim)
        self.dense = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state):
        # x: (batch, seqlen)
        enc_outs, hidden = state
        # enc_outs: (batch, seqlen, hid_dim)
        # hidden: (layers, batch, hid_dim)
        x = self.dropout(self.emb(x))
        # x: (batch, seqlen, emb_dim)
        c = self.attn(hidden[-1], enc_outs, enc_outs)
        # c: (batch, seqlen, hid_dim)
        x = torch.cat([x, c], dim=-1)
        # x: (batch, seqlen, emb_dim+hid_dim)
        out, hidden = self.rnn(x, hidden)
        # out: (batch, seqlen, hid_dim)
        # hidden: (layers, batch, hid_dim)
        out = self.dense(out)
        # out: (batch, seqlen, out_dim)
        return out, (enc_outs, hidden)

class Seq2SeqBahdanau(nn.Module):
    def __init__(self, src_dim, tgt_dim, **kw):
        super().__init__()
        self.encoder = EncoderBahdanau(in_dim=src_dim,
                                       emb_dim=kw.get('enc_emb_dim', 256),
                                       hid_dim=kw.get('enc_hid_dim', 512),
                                       n_layers=kw.get('n_layers', 1),
                                       dropout=kw.get('dropout', 0.0))
        self.decoder = DecoderBahdanau(out_dim=tgt_dim,
                                       emb_dim=kw.get('enc_emb_dim', 256),
                                       hid_dim=kw.get('enc_hid_dim', 512),
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
            # out: (batch, 1, out_dim)
            pred = out.argmax(2).squeeze(1)
            # pred: (batch,)
            outs.append(out)
        return torch.cat(outs, dim=1)

if __name__ == '__main__':
    seq2seq = Seq2SeqBahdanau(101, 102, n_layers=2)
    enc_x = torch.randint(101, (32, 10))
    dec_x = torch.randint(102, (32, 11))
    outs = seq2seq(enc_x, dec_x)
    print(outs.shape)
