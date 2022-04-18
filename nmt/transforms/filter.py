from nmt.transforms.transform import Transform


class FilterTooLongTransform(Transform):
    def __init__(self, src_max_len=None, tgt_max_len=None):
        super().__init__()
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def forward(self, input):
        if (len(input['src']) > self.src_max_len or
            len(input('tgt')) > self.tgt_max_len):
            return None
        return input
