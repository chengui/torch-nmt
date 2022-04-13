import torch
import random
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
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
        return o, h

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
            _, hidden = self.encoder(enc_x, enc_len)
            outs, _ = self.decoder(dec_x, hidden)
        else:
            _, hidden = self.encoder(enc_x, enc_len)
            # hidden: (layers, batch, hiddens)
            outs, pred = [], None
            x = dec_x[:, 0]
            for t in range(dec_x.shape[1]):
                if pred is None or (random.random() < teacher_ratio):
                    x = dec_x[:, t].unsqueeze(-1)
                else:
                    x = pred
                # x: (batch, 1)
                out, hidden = self.decoder(x, hidden)
                outs.append(out)
                # out: (batch, 1, dec_vocab)
                pred = out.argmax(2)
                # pred: (batch, 1)
            outs = torch.cat(outs, dim=1)
        # outs: (batch, seqlen, dec_vocab)
        return outs

if __name__ == '__main__':
    seq2seq = RNNSeq2Seq(101, 102, n_layers=2)
    enc_x, enc_len = torch.randint(101, (8, 10)), torch.randint(1, 10, (8,))
    dec_x, dec_len = torch.randint(102, (8, 11)), torch.randint(1, 10, (8,))
    enc_len, _ = torch.sort(enc_len, dim=0, descending=True)
    dec_len, _ = torch.sort(dec_len, dim=0, descending=True)
    outs = seq2seq(enc_x, enc_len, dec_x, dec_len)
    print(outs.shape)
