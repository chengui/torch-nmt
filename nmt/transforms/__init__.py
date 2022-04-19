import torch
from nmt.transforms.transform import Compose
from nmt.transforms.tensor import ToTensor
from nmt.transforms.tokenize import (
    WordTokenize,
    SpacyTokenize,
)
from nmt.transforms.filter import (
    TooLongFilter,
    TooShortFilter,
)
from nmt.transforms.vocab import (
    Tok2idxVocab,
    Idx2tokVocab,
)


TRANSFORMS = {
    'word_tokenize': WordTokenize,
    'spacy_tokenize': SpacyTokenize,
    'toolong_filter': TooLongFilter,
    'tooshort_filter': TooShortFilter,
    'tok2idx_vocab': Tok2idxVocab,
    'idx2tok_vocab': Idx2tokVocab,
    'to_tensor': ToTensor,
}

def create_transforms(vocab, pipe, params):
    vocab_transforms = []
    for p in vocab:
        transform = TRANSFORMS[p]
        param = params[p] if p in params else {}
        vocab_transforms.append(transform(**param))
    pipe_transforms = []
    for p in pipe:
        transform = TRANSFORMS[p]
        param = params[p] if p in params else {}
        pipe_transforms.append(transform(**param))
    return Compose(vocab_transforms), Compose(pipe_transforms)

def load_transforms(data_dir, splits):
    samples = []
    for split in splits:
        samples.append(torch.load(data_dir.file(f'{split}.pkl')))
    return samples

def save_transforms(data_dir, subsets, splits):
    for (split, subset) in zip(splits, subsets):
        file = data_dir.file(f'{split}.pkl')
        torch.save(subset.dict(), file)
