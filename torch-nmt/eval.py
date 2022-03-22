from torch import nn
from torch.utils.data import DataLoader
from nmt.processor.base import BaseProcessor
from nmt.model import create_model
from nmt.dataset import create_dataset
from nmt.util.bleu import bleu_score
from nmt.util.fs_util import load_checkpoint, load_vocab


def evaluate_loss(model, data_iter, pad_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    model.eval()
    test_loss = 0
    for (src, tgt) in data_iter:
        pred = model(src, tgt)
        # pred: (batch_size, seq, vocab_size)
        # tgt: (batch_size, seq)
        pred = pred.permute(0, 2, 1)
        loss = criterion(pred[:, :, 1:], tgt[:, 1:])
        test_loss += loss
    test_loss /= len(data_iter)
    return test_loss

def evaluate_bleu(model, data_iter, vocab):
    model.eval()
    pred_seq, ref_seq = [], []
    for (src, tgt) in data_iter:
        pred = model(src, tgt)
        # pred: (batch_size, seq, vocab_size)
        # tgt: (batch_size, seq)
        pred = pred.argmax(2)[:, 1:].tolist()
        pred_tokens = [vocab.token(indics, True) for indics in pred]
        pred_seq.extend(pred_tokens)
        tgt = tgt[:, 1:].tolist()
        tgt_tokens = [vocab.token(indics, True) for indics in tgt]
        ref_seq.extend(tgt_tokens)
    return bleu_score(pred_seq, ref_seq)

def evaluate(model, dataset, vocab, batch_size):
    test_iter = DataLoader(dataset=dataset, batch_size=batch_size)
    test_loss = evaluate_loss(model, test_iter, vocab.PAD_IDX)
    test_bleu = evaluate_bleu(model, test_iter, vocab)
    print(f'Test Error: loss={test_loss:>3f}, bleu={test_bleu:>3f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='tatoeba',
                        help='dataset to use')
    parser.add_argument('-m', '--model', default='rnn',
                        help='model to use')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='batch size of dataloader')
    parser.add_argument('-r', '--root-dir', default='.',
                         help='root dir of dataset')
    parser.add_argument('-t', '--trained', default='./output',
                        help='trained model')
    args = parser.parse_args()

    dataset = create_dataset(args.dataset, split='test',
                             root_dir=args.root_dir,
                             processor=BaseProcessor())
    src_vocab, tgt_vocab = load_vocab(args.trained)
    vocab_size = (len(src_vocab), len(tgt_vocab))
    model = create_model(args.model, vocab_size=vocab_size)
    if args.trained:
        load_checkpoint(f'{args.trained}/model.pt', model)
    evaluate(model, dataset, tgt_vocab, args.batch_size)
