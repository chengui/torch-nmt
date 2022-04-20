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

class PairCorpus(ParallelCorpus):
    def __init__(self, full_pair, **kw):
        data = self.read_pair(full_pair)
        super().__init__(data)

    def read_pair(self, fpath, sep='\t'):
        data = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for ln in f:
                src, tgt = ln.strip().split(sep)
                item = {'src': src, 'tgt': tgt}
                data.append(item)
        return data

class SingleCorpus(ParallelCorpus):
    def __init__(self, src_single, tgt_single, **kw):
        zipp = zip(self.read_single(src_single), self.read_single(tgt_single))
        data = [{'src': src, 'tgt': tgt} for (src, tgt) in zipp]
        super().__init__(data)

    def read_single(self, fpath):
        lns = []
        with open(fpath, 'r', encoding='utf-8') as f:
            lns = [ln.strip() for ln in f]
        return lns
