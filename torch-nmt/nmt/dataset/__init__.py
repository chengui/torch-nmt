from torch.utils.data import random_split
from nmt.dataset.tatoeba import TatoebaDataset

DATASETS = {
    'tatoeba': TatoebaDataset,
}

def split_dataset(dataset, ratios=[0.8, 0.2]):
    lengths = [int(len(dataset) * i) for i in ratios]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    return random_split(dataset, lengths=lengths)

def create_dataset(dataset, split='train', **kwargs):
    Dataset = DATASETS[dataset]
    return Dataset(split=split, **kwargs)
