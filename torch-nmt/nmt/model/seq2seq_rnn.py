import torch
import random
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          batch_first=True, dropout=dropout)

    def forward(self, X):
        # X: (batch_size, seq)
        X = self.embedding(X)
        # X: (batch_size, seq, embed_size)
        Y, state = self.rnn(X)
        # Y: [batch_size, seq, num_hiddens]
        # state: [batch_size, num_hiddens]
        return Y, state

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          batch_first=True, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, state):
        # X: (batch_size)
        X = X.unsqueeze(1)
        # X: (batch_size, 1)
        X = self.embedding(X)
        # X: (batch_size, 1, embed_size)
        Y, state = self.rnn(X, state)
        # Y: (batch_size, 1, num_hiddens)
        # state: (batch_size, num_hiddens)
        Y = self.dense(Y)
        # Y: (batch_size, 1, vocab_size)
        Y = Y.squeeze(1)
        # Y: (batch_size, vocab_size)
        return Y, state

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, teacher_ratio=0.5):
        # enc_X: (batch_size, seq)
        # dec_X: (batch_size, seq)
        vocab_size = self.decoder.vocab_size
        batch_size = dec_X.shape[0]
        seq_len = dec_X.shape[1]
        _, state = self.encoder(enc_X)
        # state: (batch_size, num_hiddens)
        outputs = [torch.zeros(batch_size, 1, vocab_size)]
        X = dec_X[:, 0]
        for t in range(1, seq_len):
            output, state = self.decoder(X, state)
            outputs.append(output.unsqueeze(1))
            pred = output.argmax(1)
            X = dec_X[:, t] if random.random() < teacher_ratio else pred
        # outputs: (batch_size, seq, vocab_size)
        return torch.cat(outputs, dim=1)
