import torch
from nmt.transforms.transform import Transform


class ToTensor(Transform):
    def __init__(self, dtype=torch.long):
        super().__init__()
        self.dtype = dtype

    def forward(self, input):
        input['src'] = torch.tensor(input['src'], dtype=self.dtype)
        input['tgt'] = torch.tensor(input['tgt'], dtype=self.dtype)
        return input
