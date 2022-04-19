

class ParallelCorpus(object):
    def __init__(self, src=[], tgt=[], **kw):
        super().__init__(**kw)
        self.state = [{'src': s, 'tgt': t} for (s, t) in zip(src, tgt)]

    def __getitem__(self, key):
        vals = []
        for it in self.state:
            vals.append(it.get(key, None))
        return vals

    def keys(self):
        if len(self.state) == 0:
            return []
        return self.state[0].keys()

    def dict(self):
        dct = {}
        for key in self.keys():
            dct[key] = self.__getitem__(key)
        return dct

    def apply(self, transforms):
        state = []
        for it in self.state:
            val = transforms(it)
            if val is not None:
                state.append(val)
        self.state = state
        return self

    def split(self, ratios):
        l = [int(r * len(self.state)) for r in ratios]
        states, off = [], 0
        for i in range(len(l)):
            if i == len(l) - 1:
                states.append(self.state[off:])
            else:
                states.append(self.state[off:off+l[i]])
        return [SubsetCorpus(state) for state in states]

class SubsetCorpus(ParallelCorpus):
    def __init__(self, state):
        super().__init__()
        self.state = state
