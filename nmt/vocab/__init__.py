from nmt.vocab.vocab import Vocab


SRC_VOCAB, TGT_VOCAB = 'src_vocab.txt', 'tgt_vocab.txt'

def build_vocab(corpus, max_size, min_freq=2, **kw):
    vocab = {'src': Vocab(), 'tgt': Vocab()}
    vocab['src'].build(corpus['src'], max_size, min_freq)
    vocab['tgt'].build(corpus['tgt'], max_size, min_freq)
    return vocab

def load_vocab(vocab_dir, src_file=SRC_VOCAB, tgt_file=TGT_VOCAB):
    src_vocab = Vocab.from_file(vocab_dir.rfile(src_file))
    tgt_vocab = Vocab.from_file(vocab_dir.rfile(tgt_file))
    return (src_vocab, tgt_vocab)

def save_vocab(vocab_dir, src_vocab, tgt_vocab):
    src_vocab.to_file(vocab_dir.file(SRC_VOCAB))
    tgt_vocab.to_file(vocab_dir.file(TGT_VOCAB))

def batch_toindex(tokens, vocab):
    if not isinstance(tokens, (tuple, list)):
        tokens = [tokens]
    tokens = [vocab[token] for token in tokens]
    return tokens

def batch_totoken(indics, vocab, pad_eos=False):
    if not isinstance(indics, (tuple, list)):
        indics = [indics]
    if len(indics) and isinstance(indics[0], int):
        if pad_eos:
            indics = indics + [vocab.EOS_IDX]
        return vocab.token(indics)
    else:
        batch = []
        for indic in indics:
            batch.append(batch_totoken(indic, vocab, pad_eos))
        return batch
