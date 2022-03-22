import torch
from nmt.util.vocab import Vocab


class Processor(object):
    def __call__(self, texts, vocab=None):
        return self.process(texts, vocab)

    def process(self, texts, vocab):
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        texts = self.preprocess(texts)
        tokens = self.tokenize(texts)
        if vocab is None:
            vocab = self.build_vocab(tokens)
        array = self.numerical(tokens, vocab)
        return array, vocab

    def preprocess(self, texts):
        raise NotImplementedError()

    def tokenize(self, texts):
        raise NotImplementedError()

    def build_vocab(self, tokens, min_freq=2):
        raise NotImplementedError()

    def numerical(self, tokens, vocab, maxlen=None):
        raise NotImplementedError()


class BaseProcessor(Processor):
    def __init__(self, max_seqlen=None, min_wordfreq=2,
                 preprocess=None, tokenize=None):
        super().__init__()
        self.max_seqlen = max_seqlen
        self.min_wordfreq = min_wordfreq
        self._preprocess = preprocess
        self._tokenize = tokenize

    def preprocess(self, texts):
        def space_replace(text):
            text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
            out, prev = [], ' '
            for ch in text:
                if prev != ' ' and ch in set(',.!?'):
                    out.append(' ' + ch)
                else:
                    out.append(ch)
                prev = ch
            return ''.join(out)
        if self._preprocess:
            _preprocess = self._preprocess
        else:
            _preprocess = space_replace
        return [_preprocess(text) for text in texts]

    def tokenize(self, texts):
        def space_split(text):
            return text.split(' ')
        if self._tokenize:
            _tokenize = self._tokenize
        else:
            _tokenize = space_split
        return [_tokenize(text) for text in texts]

    def build_vocab(self, tokens, min_freq=None):
        if min_freq is None:
            min_freq = self.min_wordfreq
        vocab = Vocab()
        vocab.build(tokens, min_freq)
        return vocab

    def numerical(self, tokens, vocab, maxlen=None):
        def pad(tokens, maxlen):
            pad_length = max(0, maxlen-len(tokens))
            pad_tokens = [Vocab.SOS_TOK] + tokens + [Vocab.EOS_TOK]
            pad_tokens = tokens + [Vocab.PAD_TOK] * pad_length
            return pad_tokens[:maxlen]
        if maxlen is None:
            if self.max_seqlen:
                maxlen = self.max_seqlen
            else:
                maxlen = max(len(l) for l in tokens)
        tokens = [pad(words, maxlen+2) for words in tokens]
        vector = [vocab[words] for words in tokens]
        return torch.tensor(vector)
