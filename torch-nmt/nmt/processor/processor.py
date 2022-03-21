import torch
from .vocab import Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


class Processor(object):
    def __init__(self, preprocess=None, tokenize=None):
        self.preprocess = preprocess if preprocess else self._preprocess
        self.tokenize = tokenize if tokenize else self._tokenize

    def _preprocess(self, texts):
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
        return [space_replace(text) for text in texts]

    def _tokenize(self, texts):
        def space_split(text):
            return text.split(' ')
        return [space_split(text) for text in texts]

    def build_vocab(self, tokens, min_freq=2):
        reverse_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        return Vocab(tokens, min_freq, reverse_tokens)

    def numerical(self, tokens, vocab, maxlen=None):
        def pad(tokens, maxlen):
            pad_length = max(0, maxlen-len(tokens))
            pad_tokens = tokens[:maxlen] + [PAD_TOKEN] * pad_length
            return [SOS_TOKEN] + pad_tokens + [EOS_TOKEN]
        if not maxlen:
            maxlen = max(len(l) for l in tokens)
        tokens = [pad(words, maxlen) for words in tokens]
        vector = [vocab[words] for words in tokens]
        return torch.tensor(vector)

    def process(self, texts):
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        texts = self.preprocess(texts)
        tokens = self.tokenize(texts)
        vocab = self.build_vocab(tokens)
        array = self.numerical(tokens, vocab)
        return array, vocab

    def __call__(self, texts):
        return self.process(texts)
