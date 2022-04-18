from torch.utils.data import Dataset


class ParallelCorpus(Dataset):
    def __init__(self, src, tgt, **kw):
        super().__init__(**kw)
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return (self.src[i], self.tgt[i])

    def __len__(self):
        return len(self.src)


