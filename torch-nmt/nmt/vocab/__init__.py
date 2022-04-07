import os
from nmt.vocab.vocab import Vocab


def load_vocab(data_dir, src_file='src_vocab.txt', tgt_file='tgt_vocab.txt'):
    vocab_dir = os.path.join(data_dir, 'vocab')
    if not os.path.exists(vocab_dir):
        vocab_dir = data_dir
    src_file = os.path.join(vocab_dir, src_file)
    if not os.path.exists(src_file):
        raise OSError(f'source vocab not exists: {vocab_dir}')
    tgt_file = os.path.join(vocab_dir, tgt_file)
    if not os.path.exists(tgt_file):
        raise OSError(f'target vocab not exists: {vocab_dir}')
    src_vocab = Vocab.from_file(src_file)
    tgt_vocab = Vocab.from_file(tgt_file)
    return src_vocab, tgt_vocab

def save_vocab(src_vocab, tgt_vocab, data_dir='.'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    src_file = os.path.join(data_dir, 'src_vocab.txt')
    src_vocab.to_file(src_file)
    tgt_file = os.path.join(data_dir, 'tgt_vocab.txt')
    tgt_vocab.to_file(tgt_file)
