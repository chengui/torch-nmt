
class Transform(object):
    def __call__(self, input):
        return self.forward(input)

    def warmup(self, vocab):
        self.vocab = vocab

    def get_vocab(self):
        return self.vocab

    def apply(self, inputs):
        outputs = []
        for input in inputs:
            output = self.forward(input)
            if output is not None:
                outputs.append(output)
        return outputs

    def forward(self):
        raise NotImplementedError()

class Compose(Transform):
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms = transforms

    def warmup(self, vocab):
        for transform in self.transforms:
            transform.warmup(vocab)

    def forward(self, input):
        for transform in self.transforms:
            input = transform(input)
            if input is None:
                break
        return input
