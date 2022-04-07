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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', default='./data',
                        help='')
    parser.add_argument('-l', '--max-len', default=10,
                        help="max sequence length")
    args = parser.parse_args()

    toy_dir = os.path.join(args.data_dir, 'toy')
    if not os.path.exists(toy_dir):
        os.makedirs(toy_dir)

    gen_data(toy_dir, 'train', 10000, args.max_len)
    gen_data(toy_dir, 'valid', 1000, args.max_len)
    gen_data(toy_dir, 'test', 1000, args.max_len)
