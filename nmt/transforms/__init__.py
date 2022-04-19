import torch
from nmt.transforms.tokenize import (
    WordTokenize,
    SpacyTokenize,
)
from nmt.transforms.filter import (
    TooLongFilter,
)
from nmt.transforms.vocab import (
    Tok2idxVocab,
    Idx2tokVocab,
)
from nmt.transforms.transform import (
    ToTensor,
    Compose,
)


TRANSFORMS = {
    'word_tokenize': WordTokenize,
    'spacy_tokenize': SpacyTokenize,
    'toolong_filter': TooLongFilter,
    'tok2idx_vocab': Tok2idxVocab,
    'idx2tok_vocab': Idx2tokVocab,
    'to_tensor': ToTensor,
}

def create_transforms(vocab, pipe, params):
    transforms = []
    for p in pipe:
        transform = TRANSFORMS[p]
        param = params[p] if p in params else {}
        transforms.append(transform(**param))
    return Compose(vocab, transforms)

def load_transforms(data_dir, splits):
    samples = []
    for split in splits:
        samples.append(torch.load(data_dir.file(f'{split}.pkl')))
    return samples

def save_transforms(data_dir, samples, splits):
    for (split, sample) in zip(splits, samples):
        file = data_dir.wfile(f'{split}.pkl')
        torch.save(samples, file)
