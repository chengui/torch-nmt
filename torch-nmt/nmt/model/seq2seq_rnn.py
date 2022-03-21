import torch
import random
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        # X: [seq_len, batch_size]
        X = self.embedding(X)
        # X: [seq_len, batch_size, embed_size]
        Y, state = self.rnn(X)
        # Y: [seq_len, batch_size, num_hiddens]
        # state: [batch_size, num_hiddens]
        return Y, state

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, state):
        # X: [batch_size]
        X = X.unsqueeze(0)
        # X: [1, batch_size]
        X = self.embedding(X)
        # X: [1, batch_size, embed_size]
        Y, state = self.rnn(X, state)
        # Y: [1, batch_size, num_hiddens]
        # state: [batch_size, num_hiddens]
        Y = self.dense(Y)
        # Y: [1, batch_size, vocab_size]
        Y = Y.squeeze(0)
        # Y: [batch_size, vocab_size]
        return Y, state

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, teacher_ratio=1.0):
        # enc_X: [batch_size, seq_len]
        # dec_X: [batch_size, seq_len]
        enc_X = enc_X.permute(1, 0)
        dec_X = dec_X.permute(1, 0)
        # enc_X: [seq_len, batch_size]
        # dec_X: [seq_len, batch_size]
        batch_size = dec_X.shape[1]
        seq_len = dec_X.shape[0]
        # enc_X: [seq_len, batch_size]
        _, state = self.encoder(enc_X)
        # state: [batch_size, num_hiddens]
        outputs = torch.zeros(seq_len, batch_size, self.decoder.vocab_size)
        X = dec_X[0, :]
        for t in range(1, seq_len):
            output, state = self.decoder(X, state)
            outputs[t] = output
            pred = output.argmax(1)
            X = dec_X[t] if random.random() < teacher_ratio else pred
        # outputs: [seq_len, batch_size, vocab_size]
        return outputs
