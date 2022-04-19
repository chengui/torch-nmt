from nmt.transforms.transform import Transform


class TooLongFilter(Transform):
    def __init__(self, max_srclen=None, max_tgtlen=None):
        super().__init__()
        self.max_srclen = max_srclen
        self.max_tgtlen = max_tgtlen

    def forward(self, input):
        if (len(input['src']) > self.max_srclen or
            len(input('tgt')) > self.max_tgtlen):
            return None
        return input
