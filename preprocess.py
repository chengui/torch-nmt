import torch
from nmt.transforms import create_transforms


def split_length(l, ratio):
    pass

def preprocess(corpus, pipes, params, data_dir):
    if isinstance(pipes, (list, tuple)):
        src_pipe, tgt_pipe = pipes
    else:
        src_pipe, tgt_pipe = pipes, pipes
    src_transforms = create_transforms(src_pipe, params)
    tgt_transforms = create_transforms(tgt_pipe, params)
    src_samples, tgt_samples = [], []
    for text in corpus:
        src_samples.append(src_transform(text[0]))
        tgt_samples.append(tgt_transform(text[1]))
    lens = split_length(len(src_samples), [8, 1, 1])
    for i, sp in enumerate(['train', 'valid', 'test']):
        if i == 0:
            beg, end = 0, lens[i]
        else:
            beg, end = lens[i-1]+1, lens[i]
        state_dict = {
            'src': src_samples[beg:end],
            'tgt': tgt_samples[beg:end],
        }
        torch.save(state_dict, data_dir.file(f'{sp}.pkl'))


if __name__ == '__main__':
    pass

