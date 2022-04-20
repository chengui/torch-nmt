from nmt.corpus.parallel import ParallelCorpus


class DelimiterCorpus(ParallelCorpus):
    def __init__(self, merge_file, **kw):
        data = self.read_tsv(merge_file)
        super().__init__(data)

    def read_tsv(self, fpath):
        data = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for ln in f:
                src, tgt = ln.strip().split('\t')
                item = {'src': src, 'tgt': tgt}
                data.append(item)
        return data
