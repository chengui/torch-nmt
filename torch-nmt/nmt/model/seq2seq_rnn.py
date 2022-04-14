import torch
import random
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)
from .beam_decoder import (
    beam_initial,
    beam_search
)


class RNNEncoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, l):
        # x: (batch, seqlen), l: (batch,)
        e = self.dropout(self.emb(x))
        # e: (batch, seqlen, n_embed)
        e = pack_padded_sequence(e, l.cpu(), batch_first=True,
                                 enforce_sorted=False)
        o, h = self.rnn(e)
        o, _ = pad_packed_sequence(o, batch_first=True)
        # o: (batch, seqlen, hiddens)
        # h: (layers, batch, hiddens)
        return h

class RNNDecoder(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(n_hiddens, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        # x: (batch, seqlen)
        # hidden: (layers, batch, hiddens)
        x = self.dropout(self.emb(x))
        # x: (batch, seqlen, embed)
        o, hidden = self.rnn(x, hidden)
        # o: (batch, seqlen, hiddens)
        o = self.fc(o)
        # o: (batch, seqlen, vocab)
        return o, hidden

class RNNSeq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = RNNEncoder(n_vocab=enc_vocab,
                                  n_embed=kw.get('enc_embed', 32),
                                  n_hiddens=kw.get('n_hiddens', 64),
                                  n_layers=kw.get('n_layers', 1),
                                  dropout=kw.get('dropout', 0.0))
        self.decoder = RNNDecoder(n_vocab=dec_vocab,
                                  n_embed=kw.get('dec_embed', 32),
                                  n_hiddens=kw.get('n_hiddens', 64),
                                  n_layers=kw.get('n_layers', 1),
                                  dropout=kw.get('dropout', 0.0))

    def forward(self, enc_x, enc_len, dec_x, dec_len, teacher_ratio=0.5):
        # enc_x: (batch, seqlen), enc_len: (batch,)
        # dec_x: (batch, seqlen), dec_len: (batch,)
        if teacher_ratio >= 1:
            state = self.encoder(enc_x, enc_len)
            outs, _ = self.decoder(dec_x, state)
        else:
            state = self.encoder(enc_x, enc_len)
            # state: (layers, batch, hiddens)
            outs, pred = [], None
            x = dec_x[:, 0]
            for t in range(dec_x.shape[1]):
                if pred is None or (random.random() < teacher_ratio):
                    x = dec_x[:, t].unsqueeze(-1)
                else:
                    x = pred
                # x: (batch, 1)
                out, state = self.decoder(x, state)
                outs.append(out)
                # out: (batch, 1, dec_vocab)
                pred = out.argmax(2)
                # pred: (batch, 1)
            outs = torch.cat(outs, dim=1)
        # outs: (batch, seqlen, dec_vocab)
        return outs

    @torch.no_grad()
    def predict(self, enc_x, enc_len, dec_x, dec_len, eos_idx=3, maxlen=100):
        state = self.encoder(enc_x, enc_len)
        # state: (layers, batch, hiddens)
        preds, pred = [], None
        pred_lens = maxlen * torch.ones(dec_x.shape[0]).long()
        # pred_lens: (batch,)
        for t in range(maxlen):
            x = dec_x[:, t].unsqueeze(1) if pred is None else pred
            # x: (batch, 1)
            out, state = self.decoder(x, state)
            # out: (batch, 1, dec_vocab)
            pred = out.argmax(2)
            preds.append(pred)
            # pred: (batch, 1)
            if all(pred_lens.le(t)):
                break
            pred_lens[pred_lens.gt(t) & pred.squeeze(1).eq(eos_idx)] = t
        return torch.cat(preds, dim=-1), pred_lens

    @torch.no_grad()
    def beam_predict(self, enc_x, enc_len, dec_x, dec_len,
                     beam=2, eos_idx=3, maxlen=100):
        state = self.encoder(enc_x, enc_len)
        # state: (layers, batch*beam, hiddens)
        pred = None
        pred_lens = maxlen * torch.ones(dec_x.shape[0]*beam).long()
        for t in range(maxlen):
            if t == 0:
                x = dec_x[:, t].unsqueeze(1)
            else:
                x = pred.view(-1, 1)
            # x: (batch*beam, 1)
            out, state = self.decoder(x, state)
            # out: (batch*beam, 1, dec_vocab)
            if t == 0:
                preds, scores = beam_initial(out, beam)
                state = state.repeat(1, beam, 1)
            else:
                preds, scores = beam_search(out, preds, scores, beam)
            # preds: (batch, beam, t), scores: (batch, beam)
            pred = preds[:, :, -1]
            # pred: (batch, beam)
            if all(pred_lens.le(t)):
                break
            pred_lens[pred_lens.gt(t) & pred.view(-1).eq(eos_idx)] = t
        # (batch, beam, seqlen), (batch, beam)
        return preds, pred_lens.view(-1, beam)


if __name__ == '__main__':
    seq2seq = RNNSeq2Seq(101, 102, n_layers=2)
    enc_x, enc_len = torch.randint(101, (8, 10)), torch.randint(1, 10, (8,))
    dec_x, dec_len = torch.randint(102, (8, 11)), torch.randint(1, 10, (8,))
    enc_len, _ = torch.sort(enc_len, dim=0, descending=True)
    dec_len, _ = torch.sort(dec_len, dim=0, descending=True)
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
