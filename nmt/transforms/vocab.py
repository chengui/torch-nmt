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
