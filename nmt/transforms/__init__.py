from nmt.transforms.tokenize import (
    WordTokenizeTransform,
    SpacyTokenizeTransform,
)
from nmt.transforms.filter import (
    FilterTooLongTransform,
)
from nmt.transforms.vocab import (
    VocabTok2idxTransform,
)
from nmt.transforms.transform import (
    ToTensor,
    Compose,
)


TRANSFORMS = {
    'word_tokenize': WordTokenizeTransform,
    'spacy_tokenize': SpacyTokenizeTransform,
    'filter_too_long': FilterTooLongTransform,
    'vocab_tok2idx': VocabTok2idxTransform,
    'to_tensor': ToTensor,
}

def load_transforms(pipe, params):
    compose = []
    for p in pipe:
        transform = TRANSFORMS[p]
        if p in params:
            param = params[p]
            compose.append(transform(**param))
        else:
            compose.append(transform())
    return Compose(compose)
