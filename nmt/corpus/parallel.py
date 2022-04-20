from torch.utils.data import Dataset


class ParallelCorpus(Dataset):
    def __init__(self, data, **kw):
        super().__init__(**kw)
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def keys(self):
        if len(self.data) == 0:
            return []
        return self.data[0].keys()

    def values(self, key):
        vals = []
        for it in self.data:
            vals.append(it.get(key, None))
        return vals

    def dict(self):
        dct = {}
        for key in self.keys():
            dct[key] = self.values(key)
        return dct

    def apply(self, transforms):
        data = []
        for it in self.data:
            val = transforms(it)
            if val is not None:
                data .append(val)
        self.data = data
        return self

    def split(self, ratios):
        l = [int(r * len(self.data)) for r in ratios]
        splits, off = [], 0
        for i in range(len(l)):
            if i == len(l) - 1:
                splits.append(self.data[off:])
            else:
                splits.append(self.data[off:off+l[i]])
        return [SubsetCorpus(sp) for sp in splits]

class SubsetCorpus(ParallelCorpus):
    def __init__(self, data):
        super().__init__()
        self.data = data
