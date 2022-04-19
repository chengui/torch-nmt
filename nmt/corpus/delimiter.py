from nmt.corpus.parallel import ParallelCorpus


class DelimiterCorpus(ParallelCorpus):
    def __init__(self, merge_file, **kw):
        src, tgt = self.read_tsv(merge_file)
        super().__init__(src, tgt)

    def read_tsv(self, fpath):
        src, tgt = [], []
        with open(fpath, 'r', encoding='utf-8') as f:
            for ln in f:
                s, t = ln.strip().split('\t')
                src.append(s)
                tgt.append(t)
        return src, tgt
