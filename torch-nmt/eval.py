import torch
from torch import nn
from torch.utils.data import DataLoader
from nmt.dataset import create_dataset
from nmt.vocab import load_vocab
from nmt.util import bleu_score
from nmt.model import (
    create_model,
    load_ckpt,
)


def batch_totoken(indics, vocab):
    if isinstance(indics, torch.Tensor):
        indics = indics.tolist()
    filtered = lambda i: i not in (vocab.PAD_IDX, vocab.SOS_IDX)
    batch = []
    for sent in indics:
        sent = list(filter(filtered, sent))
        if vocab.EOS_IDX in sent:
            i = sent.index(vocab.EOS_IDX)
            sent = sent[:i+1]
        batch.append(vocab.token(sent))
    return batch

def evaluate_loss(model, data_iter, criterion):
    model.eval()
    test_loss = 0
    for _, batch in enumerate(data_iter):
        src, src_len, tgt, tgt_len = batch
        tgt, gold = tgt[:, :-1], tgt[:, 1:]
        pred = model(src, src_len, tgt, tgt_len, teacher_ratio=0)
        # pred: (batch_size, seq, vocab_size)
        # gold: (batch_size, seq)
        pred = pred.permute(0, 2, 1)
        loss = criterion(pred, gold)
        test_loss += loss
    test_loss /= len(data_iter)
    return test_loss

def evaluate_bleu(model, data_iter, vocab, max_len=10):
    model.eval()
    cnd_seq, ref_seq = [], []
    for _, batch in enumerate(data_iter):
        src, src_len, tgt, tgt_len = batch
        sos = torch.full((src.shape[0], max_len), vocab.SOS_IDX)
        sos_len = torch.ones((src.shape[0],))
        pred = model(src, src_len, sos, sos_len, teacher_ratio=0)
        cnd_seq.extend(batch_totoken(pred.argmax(2), vocab))
        ref_seq.extend(batch_totoken(tgt[:, 1:], vocab))
    return bleu_score(cnd_seq, ref_seq)

def evaluate(model, dataset, vocab, batch_size=32, max_len=10):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
    test_iter = DataLoader(dataset=dataset, batch_size=batch_size)
    test_loss = evaluate_loss(model, test_iter, criterion)
    test_bleu = evaluate_bleu(model, test_iter, vocab, max_len)
    print(f'Test Error: loss={test_loss:>3f}, bleu={test_bleu:>3f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-type', required=True,
                        help='model type to use')
    parser.add_argument('-w', '--work-dir', required=True,
                        help='working dir to perform')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size of dataloader')
    parser.add_argument('-l', '--max-len', type=int, default=10,
                        help='maxium length to predict')
    args = parser.parse_args()

    src_vocab, tgt_vocab = load_vocab(args.work_dir)
    test_set, = create_dataset(args.work_dir,
                               vocab=(src_vocab, tgt_vocab),
                               split=('test',))
    model = create_model(model_type=args.model_type,
                         enc_vocab=len(src_vocab),
                         dec_vocab=len(tgt_vocab))

    load_ckpt(model, args.work_dir, mode='best')
    evaluate(model, test_set, tgt_vocab, args.batch_size, args.max_len)
