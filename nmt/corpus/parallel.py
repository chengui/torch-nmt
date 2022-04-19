from torch.utils.data import Dataset


class ParallelCorpus(Dataset):
    def __init__(self, src, tgt, **kw):
        super().__init__(**kw)
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return {
            'src': self.src[i],
            'tgt': self.tgt[i],
        }

    def __len__(self):
        return len(self.src)
