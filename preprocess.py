from nmt.corpus import create_corpus
from nmt.workdir import WorkDir
from nmt.config import Config
from nmt.transforms import (
    create_transforms,
    save_transforms,
)
from nmt.vocab import (
    save_vocab,
    build_vocab,
)


def preprocess(corpus, transforms, vocab_opts, splits, ratios, work_dir):
    vocab_transforms, pipe_transforms = create_transforms(**transforms)
    corpus = corpus.apply(vocab_transforms)
    vocab = build_vocab(corpus, **vocab_opts)
    save_vocab(work_dir.vocab, vocab['src'], vocab['tgt'])

    pipe_transforms.warmup(vocab)
    corpus = corpus.apply(pipe_transforms)
    subsets = corpus.split(ratios)
    save_transforms(work_dir.data, subsets, splits)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='configure file for model')
    parser.add_argument('-w', '--work-dir', required=True,
                        help='working dir to perform')
    parser.add_argument('-l', '--lang-pair', required=True,
                        help='language pair')
    parser.add_argument('-o', '--corpus-type', default='delimiter',
                        help='corpus type to use')
    parser.add_argument('-p', '--splits', default='train,valid,test',
                        help='splits to generate')
    parser.add_argument('-r', '--ratios', default='0.8,0.1,0.1',
                        help='splits ratio to use')
    args = parser.parse_args()

    wdir = WorkDir(args.work_dir)
    conf = Config.load_config(args.config)

    splits = args.splits.split(',')
    lang_pair = tuple(args.lang_pair.split('-'))
    ratios = list(map(float, args.ratios.split(',')))

    corpus = create_corpus(wdir.corpus, lang_pair, args.corpus_type)
    preprocess(corpus, conf.transforms, conf.vocab, splits, ratios, wdir)
