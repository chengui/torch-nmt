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

class AddLen(Transform):
    def forward(self, input):
        input['src_len'] = torch.tensor(len(input['src'])).long()
        input['tgt_len'] = torch.tensor(len(input['tgt'])).long()
        return input
