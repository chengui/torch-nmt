from nmt.dataset.pickle import PickleDataset

DATASETS = {
    'pickle': PickleDataset,
}

def create_dataset(data_dir, split=('train',), ftype='pickle'):
    Dataset = DATASETS.get(ftype, None)
    if not Dataset:
        raise ValueError(f'unsupported dataset format: {ftype}')

    paths = [data_dir.rfile(f'{sp}.pkl') for sp in split]
    return [Dataset(path) for path in paths]
