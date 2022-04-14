import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class LuongAttention(nn.Module):
    def __init__(self, n_hiddens, score_fn='dot'):
        super().__init__()
        self.score_fn = score_fn
        if score_fn == 'general':
            self.w_a = nn.Linear(n_hiddens, n_hiddens)
        elif score_fn == 'concat':
            self.w_a = nn.Linear(n_hiddens*2, n_hiddens)
            self.v_a = torch.FloatTensor(n_hiddens)

    def forward(self, q, k, v, mask=None):
        # q: (batch, hiddens)
        # k: (batch, seqlen, hiddens)
        # v: (batch, seqlen, hiddens)
        e = getattr(self, self.score_fn)(q, k)
        # e: (batch, 1, seqlen)
        if mask is not None:
            e = e.masked_fill(mask==0, -1e10)
        a = F.softmax(e, dim=-1)
        # a: (batch, 1, seqlen)
        c = torch.bmm(a, v)
        # c: (batch, 1, hiddens)
        return c

    def dot(self, q, k):
        return torch.bmm(q.unsqueeze(1), k.permute(0, 2, 1))

    def general(self, q, k):
        e = self.w_a(q.unsqueeze(1))
        return torch.bmm(e, k.permute(0, 2, 1))

    def concat(self, q, k):
        q = q.unsqueeze(1).repeat(1, k.shape[1], 1)
        cat = torch.cat([q, k], dim=-1)
        # cat: (batch, seqlen, hiddens*2)
        w = torch.tanh(self.w_a(cat))
        # w: (batch, seqlen, hiddens)
        return torch.sum(self.v_a.to(w.device)*w, dim=-1).unsqueeze(1)

class LuongEncoder(nn.Module):
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
            _, bs, nh = h.shape
            h = h.view(-1, 2, bs, nh)
            # h: (layers, 2, batch, hiddens)
            h = torch.stack([torch.cat((i[0], i[1]), dim=1) for i in h])
        # h: (layers, batch, hiddens*dir)
        return (o, h)

class LuongDecoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1,
                 score_fn='dot', use_birnn=False):
        super().__init__()
        if use_birnn: n_hiddens *= 2
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)
        self.att = LuongAttention(n_hiddens, score_fn)
        self.out = nn.Linear(n_hiddens*2, n_vocab)
        ## self.out = nn.Linear(n_embed+n_hiddens*2, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, state, mask):
        # y: (batch, seqlen)
        ctx, s = state
        # ctx: (batch, seqlen, hiddens)
        # s: (layers, batch, hiddens)
        e = self.dropout(self.emb(y))
        # e: (batch, seqlen, embed)
        o, s = self.rnn(e, s)
        # o: (batch, seqlen, hiddens)
        # s: (layers, batch, hiddens)
        c = self.att(s[-1], ctx, ctx, mask)
        # c: (batch, 1, hiddens)
        ## cat = torch.cat([e, c, o], dim=-1)
        # cat: (batch, 1, embed+hiddens*2)
        ## o = self.out(cat)
        # o: (batch, 1, vocab)
        s_ = s[-1].unsqueeze(1)
        cat = torch.cat([s_, c], dim=-1)
        # cat: (batch, 1, hiddens*2)
        o = self.out(cat)
        # o: (batch, 1, vocab)
        return o, (ctx, s)

class LuongSeq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = LuongEncoder(n_vocab=enc_vocab,
                                    n_embed=kw.get('enc_embed', 256),
                                    n_hiddens=kw.get('n_hiddens', 512),
                                    n_layers=kw.get('n_layers', 1),
                                    use_birnn=kw.get('use_birnn', False),
                                    dropout=kw.get('dropout', 0.0))
        self.decoder = LuongDecoder(n_vocab=dec_vocab,
                                    n_embed=kw.get('dec_embed', 256),
                                    n_hiddens=kw.get('n_hiddens', 512),
                                    n_layers=kw.get('n_layers', 1),
                                    score_fn=kw.get('score_fn', 'dot'),
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
        if teacher_ratio >= 1:
            enc_mask = self.make_enc_mask(enc_x, enc_len)
            state = self.encoder(enc_x, enc_len)
            outs, _ = self.decoder(dec_x, state, enc_mask)
        else:
            enc_mask = self.make_enc_mask(enc_x, enc_len)
            state = self.encoder(enc_x, enc_len)
            pred, outs = None, []
            for t in range(dec_x.shape[1]):
                if pred is None or (random.random() < teacher_ratio):
                    x = dec_x[:, t].unsqueeze(-1)
                else:
                    x = pred
                # x: (batch, 1)
                out, state = self.decoder(x, state, enc_mask)
                outs.append(out)
                # out: (batch, 1, vocab)
                pred = out.argmax(2)
                # pred: (batch, 1)
            outs = torch.cat(outs, dim=1)
        return outs

    @torch.no_grad()
    def predict(self, enc_x, enc_len, dec_x, dec_len, eos_idx=3, maxlen=100):
        enc_mask = self.make_enc_mask(enc_x, enc_len)
        state = self.encoder(enc_x, enc_len)
        # state: (layers, batch, hiddens)
        preds, pred = [], None
        pred_lens = maxlen * torch.ones(dec_x.shape[0]).long()
        # pred_lens: (batch,)
        for t in range(maxlen):
            x = dec_x[:, t].unsqueeze(1) if pred is None else pred
            # x: (batch, 1)
            out, state = self.decoder(x, state, enc_mask)
            # out: (batch, 1, vocab)
            pred = out.argmax(2)
            preds.append(pred)
            # pred: (batch, 1)
            if all(pred_lens.le(t)):
                break
            pred_lens[pred_lens.gt(t) & pred.squeeze(1).eq(eos_idx)] = t
        return torch.cat(preds, dim=-1), pred_lens


if __name__ == '__main__':
    seq2seq = LuongSeq2Seq(101, 102, n_layers=2, use_birnn=True)
    enc_x, enc_len = torch.randint(101, (32, 10)), torch.randint(1, 10, (32,))
    dec_x, dec_len = torch.randint(102, (32, 11)), torch.randint(1, 10, (32,))
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
