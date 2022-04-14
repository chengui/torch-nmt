import os
from nmt.vocab.vocab import Vocab


SRC_VOCAB = os.getenv('SRC_VOCAB', 'src_vocab.txt')
TGT_VOCAB = os.getenv('TGT_VOCAB', 'tgt_vocab.txt')


def load_vocab(work_dir, src_file=SRC_VOCAB, tgt_file=TGT_VOCAB):
    vocab_dir = os.path.join(work_dir, 'vocab')
    if not os.path.exists(vocab_dir):
        raise OSError('vocab dir not exists: {work_dir}')
    src_file = os.path.join(vocab_dir, src_file)
    if not os.path.exists(src_file):
        raise OSError(f'source vocab not exists: {vocab_dir}')
    tgt_file = os.path.join(vocab_dir, tgt_file)
    if not os.path.exists(tgt_file):
        raise OSError(f'target vocab not exists: {vocab_dir}')
    src_vocab = Vocab.from_file(src_file)
    tgt_vocab = Vocab.from_file(tgt_file)
    return src_vocab, tgt_vocab

def save_vocab(src_vocab, tgt_vocab, work_dir):
    vocab_dir = os.path.join(work_dir, 'vocab')
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    src_file = os.path.join(vocab_dir, SRC_VOCAB)
    src_vocab.to_file(src_file)
    tgt_file = os.path.join(vocab_dir, TGT_VOCAB)
    tgt_vocab.to_file(tgt_file)

def batch_toindex(tokens, vocab):
    if not isinstance(tokens, (tuple, list)):
        tokens = [tokens]
    tokens = [vocab[token] for token in tokens]
    return tokens

def batch_totoken(indics, vocab, padding_eos=False):
    if not isinstance(indics, (tuple, list)):
        indics = [indics]
    if len(indics) and isinstance(indics[0], int):
        if padding_eos:
            indics = indics + [vocab.PAD_IDX]
        return vocab.token(indics)
    else:
        batch = []
        for indic in indics:
            batch.append(batch_totoken(indic, vocab, padding_eos))
        return batch
