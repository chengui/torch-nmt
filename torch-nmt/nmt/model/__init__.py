from torch import nn
from nmt.model.seq2seq_gru import Seq2SeqGRU

MODELS = {
    'gru': Seq2SeqGRU,
}

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def create_model(model, vocab_size):
    Seq2Seq = MODELS[model]
    (enc_vocab, dec_vocab) = vocab_size
    seq2seq = Seq2Seq(enc_vocab, dec_vocab)
    seq2seq.apply(init_weights)
    return seq2seq
