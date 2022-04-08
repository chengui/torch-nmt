import os
from nmt.dataset.tabular import TabularDataset

DATASETS = {
    'tabular': TabularDataset,
}

def create_dataset(work_dir, vocab, split=('train',), ftype='tabular'):
    data_dir = os.path.join(work_dir, 'data')
    if not os.path.exists(data_dir):
        raise OSError(f'data dir not exists: {work_dir}')

    if isinstance(vocab, (list, tuple)):
        src_vocab, tgt_vocab = vocab
    else:
        src_vocab, tgt_vocab = vocab, vocab

    if ftype not in DATASETS:
        raise ValueError(f'unsupported dataset format: {ftype}')

    Dataset = DATASETS[ftype]
    paths = [os.path.join(data_dir, f'{sp}.txt') for sp in split]
    return [Dataset(path, src_vocab, tgt_vocab) for path in paths]
