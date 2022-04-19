import spacy
from nmt.transforms.transform import Transform


class WordTokenize(Transform):
    def __init__(self, sep=' '):
        super().__init__()
        self.sep = sep

    def forward(self, input):
        input['src'] = input['src'].split(self.sep)
        input['tgt'] = input['tgt'].split(self.sep)
        return input

class SpacyTokenize(Transform):
    models = spacy.errors.OLD_MODEL_SHORTCUTS

    def __init__(self, src_lang, tgt_lang):
        super().__init__()
        self.src_nlp = spacy.load(self.models[src_lang])
        self.tgt_nlp = spacy.load(self.models[tgt_lang])
        self.src_tok = self.src_nlp.tokenizer
        self.tgt_tok = self.tgt_nlp.tokenizer

    def forward(self, input):
        input['src'] = [tok.text for tok in self.src_tok(input['src'])]
        input['tgt'] = [tok.text for tok in self.tgt_tok(input['tgt'])]
        return input
