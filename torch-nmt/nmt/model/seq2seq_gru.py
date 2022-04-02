import torch
import random
from torch import nn


class EncoderGRU(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # X: (batch, seqlen)
        X = self.dropout(self.emb(X))
        # X: (batch, seqlen, n_embed)
        Y, hidden = self.rnn(X)
        # Y: (batch, seqlen, n_hiddens)
        # hidden: (n_layers, batch, n_hiddens]
        return Y, hidden

class DecoderGRU(nn.Module):
    def __init__(self, n_vocab, n_embed, n_hiddens, n_layers, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embed)
        self.rnn = nn.GRU(n_embed, n_hiddens,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(n_hiddens, n_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, hidden):
        # X: (batch, seqlen)
        # hidden: (n_layers, batch, n_hiddens)
        X = self.dropout(self.emb(X))
        # X: (batch, seqlen, n_embed)
        Y, hidden = self.rnn(X, hidden)
        # Y: (batch, seqlen, n_hiddens)
        Y = self.fc(Y)
        # Y: (batch, seqlen, n_vocab)
        return Y, hidden

class Seq2SeqGRU(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, **kw):
        super().__init__()
        self.encoder = EncoderGRU(n_vocab=enc_vocab,
                                  n_embed=kw.get('enc_embed', 32),
                                  n_hiddens=kw.get('enc_hiddens', 64),
                                  n_layers=kw.get('n_layers', 1),
                                  dropout=kw.get('dropout', 0.0))
        self.decoder = DecoderGRU(n_vocab=dec_vocab,
                                  n_embed=kw.get('dec_embed', 32),
                                  n_hiddens=kw.get('dec_hiddens', 64),
                                  n_layers=kw.get('n_layers', 1),
                                  dropout=kw.get('dropout', 0.0))

    def forward(self, enc_X, dec_X, teacher_ratio=0.5):
        # enc_X: (batch, seqlen)
        # dec_X: (batch, seqlen)
        seqlen = dec_X.shape[1]
        _, hidden = self.encoder(enc_X)
        # hidden: (n_layers, batch, n_hiddens)
        outs, pred = [], None
        X = dec_X[:, 0]
        for t in range(seqlen):
            if pred is None or (random.random() < teacher_ratio):
                X = dec_X[:, t]
            else:
                X = pred
            # X: (batch,)
            out, hidden = self.decoder(X.unsqueeze(1), hidden)
            # out: (batch, 1, dec_vocab)
            pred = out.argmax(2).squeeze(1)
            outs.append(out)
        # out: (batch, seqlen, dec_vocab)
        return torch.cat(outs, dim=1)

if __name__ == '__main__':
    seq2seq = Seq2SeqGRU(101, 102, n_layers=2)
    enc_x = torch.randint(101, (32, 10))
    dec_x = torch.randint(102, (32, 11))
    outs = seq2seq(enc_x, dec_x)
    print(outs.shape)
