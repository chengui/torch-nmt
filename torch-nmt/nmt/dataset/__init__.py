from torch.utils.data import random_split
from nmt.dataset.tatoeba import TatoebaDataset

DATASETS = {
    'tatoeba': TatoebaDataset,
}

def split_dataset(dataset, lengths=[0.8, 0.2], **kwargs):
    lengths = [int(len(dataset) * i) for i in lengths]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    return random_split(dataset, lengths=lengths)

def create_dataset(dataset, split='train', **kwargs):
    Dataset = DATASETS[dataset]
    return Dataset(split=split, **kwargs)
