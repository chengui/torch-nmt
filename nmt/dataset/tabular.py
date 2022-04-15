from torch.utils.data import Dataset
from nmt.dataset.data import read_tsv, numerical


class TabularDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab, **kw):
        super().__init__(**kw)
        src_toks, tgt_toks = read_tsv(path)
        self.examples = list(zip(src_toks, tgt_toks))
        self.src_data, self.src_len = numerical(src_toks, src_vocab)
        self.tgt_data, self.tgt_len = numerical(tgt_toks, tgt_vocab)

    def __getitem__(self, i):
        return (self.src_data[i], self.src_len[i],
                self.tgt_data[i], self.tgt_len[i])

    def __len__(self):
        return len(self.examples)
