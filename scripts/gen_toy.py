import os
import random


def gen_data(root, name, size, maxlen=10):
    path = os.path.join(root, name+'.txt')
    with open(path, 'w') as wf:
        for _ in range(size):
            l = random.randint(1, maxlen)
            seq = [str(random.randint(0, 9)) for _ in range(l)]
            wf.write('%s\t%s\n' % (' '.join(seq),
                                   ' '.join(reversed(seq))))

def gen_vocab(vocab_dir):
    src_vocab = os.path.join(vocab_dir, 'src_vocab.txt')
    with open(src_vocab, 'w') as wf:
        wf.write('\n'.join(map(str, range(10))))
    tgt_vocab = os.path.join(vocab_dir, 'tgt_vocab.txt')
    with open(tgt_vocab, 'w') as wf:
        wf.write('\n'.join(map(str, range(10))))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--work-dir', default='./work',
                        help='')
    parser.add_argument('-l', '--max-len', default=10,
                        help="max sequence length")
    args = parser.parse_args()

    data_dir = os.path.join(args.work_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    gen_data(data_dir, 'train', 10000, args.max_len)
    gen_data(data_dir, 'valid', 1000, args.max_len)
    gen_data(data_dir, 'test', 1000, args.max_len)

    vocab_dir = os.path.join(args.work_dir, 'vocab')
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    gen_vocab(vocab_dir)
