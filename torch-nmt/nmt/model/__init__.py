from nmt.model.seq2seq_rnn import EncoderRNN, DecoderRNN, Seq2SeqRNN

MODELS = {
    'rnn': (EncoderRNN, DecoderRNN, Seq2SeqRNN),
}

def create_model(model, vocab_size, embed_size=32, num_hiddens=32, num_layers=2, dropout=0.1):
    (Encoder, Decoder, Seq2Seq) = MODELS[model]
    encoder = Encoder(vocab_size[0], embed_size, num_hiddens, num_layers, dropout)
    decoder = Decoder(vocab_size[1], embed_size, num_hiddens, num_layers, dropout)
    seq2seq = Seq2Seq(encoder, decoder)
    return seq2seq
