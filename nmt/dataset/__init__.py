from nmt.dataset.tabular import TabularDataset

DATASETS = {
    'tabular': TabularDataset,
}

def create_dataset(data_dir, vocab, split=('train',), ftype='tabular'):
    if isinstance(vocab, (list, tuple)):
        src_vocab, tgt_vocab = vocab
    else:
        src_vocab, tgt_vocab = vocab, vocab

    Dataset = DATASETS.get(ftype, None)
    if not Dataset:
        raise ValueError(f'unsupported dataset format: {ftype}')

    paths = [data_dir.rfile(f'{sp}.txt') for sp in split]
    return [Dataset(path, src_vocab, tgt_vocab) for path in paths]
