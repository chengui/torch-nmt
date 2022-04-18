import torch
from torch.utils.data import Dataset


class PickleDataset(Dataset):
    def __init__(self, path, **kw):
        super().__init__(**kw)
        state_dict = torch.load(path)
        self.src_data = state_dict['src']
        self.src_len = state_dict['src_len']
        self.tgt_data = state_dict['tgt']
        self.tgt_len = state_dict['tgt_len']

    def __getitem__(self, i):
        return (self.src_data[i], self.src_len[i],
                self.tgt_data[i], self.tgt_len[i])

    def __len__(self):
        return len(self.src_data)
