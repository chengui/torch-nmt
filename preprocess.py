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


def preprocess(corpus, transforms, splits, ratios, work_dir):
    corpus = transforms.apply(corpus)
    subsets = corpus.split(splits, ratios)
    save_transforms(work_dir.data, subsets, splits)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='configure file for model')
    parser.add_argument('-w', '--work-dir', required=True,
                        help='working dir to perform')
    parser.add_argument('-o', '--corpus-type', default=None,
                        help='corpus type to use')
    parser.add_argument('-s', '--src-lang', required=True,
                        help='source language')
    parser.add_argument('-t', '--tgt-lang', required=True,
                        help='target language')
    parser.add_argument('-p', '--splits', default='train,valid,test',
                        help='splits to generate')
    parser.add_argument('-r', '--ratios', default='0.8,0.1,0.1',
                        help='splits ratio to use')
    args = parser.parse_args()

    wdir = WorkDir(args.work_dir)
    conf = Config.load_config(args.config)

    splits = args.splits.split(',')
    ratios = list(map(float, args.ratios.split(',')))

    corpus = create_corpus(wdir.corpus, args.corpus_type)
    vocab = build_vocab(corpus, **conf.vocab)
    save_vocab(wdir.vocab, vocab['src'], vocab['tgt'])

    transforms = create_transforms(vocab, conf.transforms)
    preprocess(corpus, transforms, splits, ratios, wdir)
