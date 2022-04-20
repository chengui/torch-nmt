from nmt.corpus.parallel import ParallelCorpus


class SeparatedCorpus(ParallelCorpus):
    def __init__(self, src_file, tgt_file, **kw):
        src, tgt = self.read_corpus(src_file), self.read_corpus(tgt_file)
        data = [{'src': s, 'tgt': t} for (s, t) in zip(src, tgt)]
        super().__init__(data)

    def read_corpus(self, fpath):
        lns = []
        with open(fpath, 'r', encoding='utf-8') as f:
            lns = [ln.strip() for ln in f]
        return lns
