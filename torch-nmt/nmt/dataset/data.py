import torch


def read_tsv(path):
    src, tgt = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            s, t = ln.strip().split('\t')
            src.append(s.split(' '))
            tgt.append(t.split(' '))
    return src, tgt

def numerical(tokens, vocab, maxlen=None):
    full = lambda s: [vocab.SOS_IDX] + vocab[s] + [vocab.EOS_IDX]
    tokens = [full(sent) for sent in tokens]
    if maxlen is None:
        maxlen = max(len(sent) for sent in tokens)
    tokens = [pad(sent, maxlen, vocab.PAD_IDX) for sent in tokens]
    data = torch.tensor(tokens).long()
    lens = (data != vocab.PAD_IDX).sum(dim=1)
    return data, lens

def pad(sent, maxlen, pad_idx):
    sent = sent + [pad_idx] * max(0, maxlen-len(sent))
    return sent[:maxlen]
