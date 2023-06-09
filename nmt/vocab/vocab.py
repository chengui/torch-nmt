import collections


class Vocab(object):
    UNK_IDX, UNK_TOK = 0, '<unk>'
    PAD_IDX, PAD_TOK = 1, '<pad>'
    SOS_IDX, SOS_TOK = 2, '<sos>'
    EOS_IDX, EOS_TOK = 3, '<eos>'

    def __init__(self, reserved_tokens=[]):
        self.reserved_tokens = [
            Vocab.UNK_TOK,
            Vocab.PAD_TOK,
            Vocab.SOS_TOK,
            Vocab.EOS_TOK,
        ] + reserved_tokens
        self.itos = self.reserved_tokens[:]
        self.stoi = {v: k for k, v in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, tokens):
        return self.index(tokens)

    def build(self, tokens, max_size, min_freq=2):
        if isinstance(tokens[0], list):
            tokens = [tok for line in tokens for tok in line]
        freqs = collections.Counter(tokens)
        freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        for cnt, (tok, freq) in enumerate(freqs):
            if freq < min_freq or cnt >= max_size:
                break
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self

    def load(self, data):
        if isinstance(data, dict):
            self.stoi.update(data)
            self.itos = [x[0] for x in sorted(data.items(), key=lambda x: x[1])]
        else:
            self.itos += list(data)
            self.stoi.update({v: k for k, v in enumerate(self.itos)})
        return self

    def token(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.itos[indices]
        return [self.itos[i] for i in indices]

    def index(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.stoi.get(tokens, 0)
        return [self.stoi.get(tok, 0) for tok in tokens]

    def to_file(self, filename):
        with open(filename, 'w') as f:
            rl = len(self.reserved_tokens)
            f.write('\n'.join(self.itos[rl:]) + '\n')

    @classmethod
    def from_file(cls, filename):
        vocab = Vocab()
        with open(filename, 'r') as f:
            vocab.load([line.strip() for line in f.readlines()])
        return vocab
