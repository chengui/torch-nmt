from nmt.corpus.parallel import (
    PairCorpus,
    SingleCorpus,
)

CORPUS = {
    'pair':   PairCorpus,
    'single': SingleCorpus,
}

def create_corpus(corpus_dir, corpus_files):
    if 'full' in corpus_files:
        ful_files = corpus_files['full']
        ful_files = [corpus_dir.rfile(f) for f in ful_files]
        return [PairCorpus(split) for split in ful_files]
    else:
        zip_files = zip(corpus_files['source'], corpus_files['target'])
        zip_files = [corpus_dir.rfile(f) for f in zip_files]
        return [SingleCorpus(src, tgt) for (src, tgt) in zip_files]
