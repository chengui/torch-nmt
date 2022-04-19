
class Transform(object):
    def __call__(self, input):
        return self.forward(input)

    def warmup(self, vocab):
        self.vocab = vocab

    def apply(self, inputs):
        outs = []
        for input in inputs:
            out = self.forward(input)
            if out is not None:
                outs.append(out)
        return outs

    def forward(self):
        raise NotImplementedError()

class Compose(Transform):
    def __init__(self, vocab, transforms=[]):
        super().__init__()
        self.vocab = vocab
        self.transforms = transforms
        for transform in self.transforms:
            transform.warmup(vocab)

    def get_vocab(self):
        return self.vocab

    def forward(self, input):
        for transform in self.transforms:
            input = transform(input)
            if input is None:
                break
        return input
