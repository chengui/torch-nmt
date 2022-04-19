from nmt.transforms.transform import Transform


class Tok2idxVocab(Transform):
    def warmup(self, vocab):
        self.src_vocab = vocab['src']
        self.tgt_vocab = vocab['tgt']

    def forward(self, input):
        input['src'] = self.src_vocab.index(input['src'])
        input['tgt'] = self.tgt_vocab.index(input['tgt'])
        return input

class Idx2tokVocab(Transform):
    def warmup(self, vocab):
        self.src_vocab = vocab['src']
        self.tgt_vocab = vocab['tgt']

    def forward(self, input):
        input['src'] = self.src_vocab.token(input['src'])
        input['tgt'] = self.tgt_vocab.token(input['tgt'])
        return input

class BoundTokenVocab(Transform):
    def warmup(self, vocab):
        self.src_vocab = vocab['src']
        self.tgt_vocab = vocab['tgt']
        self.src_sos = self.src_vocab.SOS_IDX
        self.src_eos = self.src_vocab.EOS_IDX
        self.tgt_sos = self.tgt_vocab.SOS_IDX
        self.tgt_eos = self.tgt_vocab.EOS_IDX

    def forward(self, input):
        input['src'] = [self.src_sos] + input['src'] + [self.src_eos]
        input['tgt'] = [self.tgt_sos] + input['tgt'] + [self.tgt_eos]
        return input

class PadTokenVocab(Transform):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len

    def warmup(self, vocab):
        self.src_vocab = vocab['src']
        self.tgt_vocab = vocab['tgt']
        self.src_pad = self.src_vocab.PAD_IDX
        self.tgt_pad = self.tgt_vocab.PAD_IDX

    def forward(self, input):
        src_pad = [self.src_pad] * (self.max_len-len(input['src']))
        input['src'] = input['src'] + src_pad
        tgt_pad = [self.tgt_pad] * (self.max_len-len(input['tgt']))
        input['tgt'] = input['tgt'] + tgt_pad
        return input
