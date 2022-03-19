import os
import torch
import collections

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

class Vocab(object):
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        self.build(tokens, min_freq, reserved_tokens)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, tokens):
        return self.index(tokens)

    def build(self, tokens, min_freq, reserved_tokens):
        if isinstance(tokens[0], list):
            tokens = [tok for line in tokens for tok in line]
        self.freqs = collections.Counter(tokens)
        self.itos = ['<unk>'] + reserved_tokens
        self.stoi = collections.defaultdict(int)
        freqs = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        for tok, freq in freqs:
            if freq < min_freq:
                break
            self.itos.append(tok)
        self.stoi.update({v: k for k, v in enumerate(self.itos)})
        return self

    def load(self, data):
        if isinstance(data, dict):
            self.stoi = data
            self.itos = [x[0] for x in sorted(data.items(), key=lambda x: x[1])]
        else:
            self.itos = list(data)
            self.stoi.update({v: k for k, v in enumerate(self.itos)})
        return self

    def token(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.long().tolist()
        if not isinstance(indices, (list, tuple)):
            return self.itos[indices]
        return [self.itos[index] for index in indices]

    def index(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.stoi[tokens]
        return [self.stoi[tok] for tok in tokens]

def load_vocab(self, indir):
    print(f'Load vocab from {indir}...')
    src_vocab = Vocab()
    with open(os.path.join(indir, 'src_vocab.txt'), 'r') as f:
        data = f.readlines()
        data = [line.strip().split('\t') for line in data]
        src_vocab.load(dict(data))
    tgt_vocab = Vocab()
    with open(os.path.join(indir, 'tgt_vocab.txt'), 'r') as f:
        data = f.readlines()
        data = [line.strip().split('\t') for line in data]
        tgt_vocab.load(dict(data))
    return src_vocab, tgt_vocab

def save_vocab(self, outdir, src_vocab, tgt_vocab):
    with open(os.path.join(outdir, 'src_vocab.txt'), 'w') as f:
        for k, v in enumerate(src_vocab.itos):
            f.write(f'{v}\t{k}')
    with open(os.path.join(outdir, 'tgt_vocab.txt'), 'w') as f:
        for k, v in enumerate(tgt_vocab.itos):
            f.write(f'{v}\t{k}')
    print(f'Save vocab to {outdir}...')
