from nmt.dataset.delimiter import DelimiterCorpus
from nmt.dataset.separated import SeparatedCorpus

CORPUS = {
    'delimiter': DelimiterCorpus,
    'separated': SeparatedCorpus,
}

def create_corpus(corpus_dir, lang_pair, corpus_type='delimiter'):
    Corpus = CORPUS.get(corpus_type, None)
    if not Corpus:
        raise ValueError(f'unsupported corpus format: {corpus_type}')
    if corpus_type == 'delimiter':
        merge_file = f'{lang_pair[0]}-{lang_pair[1]}.txt'
        return Corpus(corpus_dir.rfile(merge_file))
    else:
        src_file, tgt_file = f'{lang_pair[0]}.txt', f'{lang_pair[1]}.txt'
        return Corpus(corpus_dir.rfile(src_file), corpus_dir.rfile(tgt_file))
