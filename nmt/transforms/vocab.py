from nmt.transforms.transforms import Transform


class VocabTok2idxTransform(Transform):
    def __init__(self, vocab=None):
        super().__init__()
        self.vocab = vocab

    def forward(self, input):
        return self.vocab[input]
