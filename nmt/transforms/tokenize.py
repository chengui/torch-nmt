import spacy
from nmt.transforms.transform import Transform


class WordTokenizeTransform(Transform):
    def __init__(self, sep=' '):
        super().__init__()
        self.sep = sep

    def forward(self, input):
        return input.split(self.sep)

class SpacyTokenizeTransform(Transform):
    models = {
        'en': 'en_core_news_sm',
        'de': 'en_core_web_sm',
    }

    def __init__(self, lang='en'):
        super().__init__()
        if lang not in self.models:
            raise KeyError(f'unsupported language: {lang}')
        self.nlp = spacy.load(self.models[lang])

    def forward(self, input):
        return [tok.text for tok in self.nlp.tokenizer(input)]
