import torch


def read_tsv(path):
    src, tgt = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            s, t = ln.strip().split('\t')
            src.append(s.split(' '))
            tgt.append(t.split(' '))
    return src, tgt

def tolist(x, x_len=None):
    if x_len is None:
        return x.tolist()
    s = []
    for i in range(x.shape[0]):
        si = []
        for j in range(x.shape[1]):
            si.append(x[i,j,:x_len[i,j]].tolist())
        s.append(si)
    return s

def pad(sent, maxlen, pad_idx):
    sent = sent + [pad_idx] * max(0, maxlen-len(sent))
    return sent[:maxlen]

def numerical(tokens, vocab, maxlen=None, device=None):
    full = lambda s: [vocab.SOS_IDX] + vocab[s] + [vocab.EOS_IDX]
    tokens = [full(sent) for sent in tokens]
    if maxlen is None:
        maxlen = max(len(sent) for sent in tokens)
    tokens = [pad(sent, maxlen, vocab.PAD_IDX) for sent in tokens]
    data = torch.LongTensor(tokens)
    lens = (data != vocab.PAD_IDX).sum(dim=1)
    if device is None:
        return data, lens
    return data.to(device), lens.to(device)

def init_target(batch_size, vocab, maxlen, device=None):
    sos_idx, pad_idx = vocab.SOS_IDX, vocab.PAD_IDX
    sos = [sos_idx] + [pad_idx]*(maxlen-1)
    sos = torch.LongTensor([sos]).repeat(batch_size, 1)
    sos_len = torch.LongTensor([1]).repeat(batch_size)
    if device is None:
        return sos, sos_len
    return sos.to(device), sos_len.to(device)
