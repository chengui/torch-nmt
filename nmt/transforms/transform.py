import torch


class Transform(object):
    def __call__(self, input):
        return self.forward(input)

    def forward(self):
        raise NotImplementedError()

class Compose(Transform):
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms = transforms

    def forward(self, input):
        for transform in self.transforms:
            input = transform(input)
        return input

class ToTensor(Transform):
    def __init__(self, dtype=torch.long):
        super().__init__()
        self.dtype = dtype

    def forward(self, input):
        return torch.tensor(input, dtype=self.dtype)
