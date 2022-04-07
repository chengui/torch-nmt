import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from nmt.dataset import create_dataset
from nmt.vocab import load_vocab
from nmt.util import clip_grad
from nmt.model import (
    create_model,
    load_ckpt,
    save_ckpt,
)


def valid_epoch(model, data_iter, criterion):
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for idx, batch in enumerate(data_iter):
            src, src_len, tgt, tgt_len = batch
            tgt, gold = tgt[:, :-1], tgt[:, 1:]
            pred = model(src, src_len, tgt, tgt_len)
            # pred: (batch, seqlen, vocab)
            # gold: (batch, seq)
            pred = pred.permute(0, 2, 1)
            loss = criterion(pred, gold)
            valid_loss += loss
        valid_loss /= len(data_iter)
        return valid_loss

def train_epoch(model, data_iter, criterion, optimizer):
    model.train()
    train_loss = 0
    for idx, batch in enumerate(data_iter):
        src, src_len, tgt, tgt_len = batch
        tgt, gold = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        pred = model(src, src_len, tgt, tgt_len)
        # pred: (batch, seqlen, vocab)
        # gold: (batch, seqlen)
        pred = pred.permute(0, 2, 1)
        loss = criterion(pred, gold)
        loss.backward()
        # clip grad
        clip_grad(model, 1)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(data_iter)
    return train_loss

def train(model, train_set, valid_set, src_vocab, tgt_vocab, work_dir=None,
          num_epochs=10, batch_size=32, learning_rate=0.001):
    train_iter = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            shuffle=True)
    valid_iter = DataLoader(dataset=valid_set,
                            batch_size=batch_size,
                            shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_hist, valid_hist = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_iter, criterion, optimizer)
        valid_loss = valid_epoch(model, valid_iter, criterion)

        train_hist.append(train_loss)
        valid_hist.append(valid_loss)

        if work_dir is not None:
            save_ckpt(work_dir, model, mode='last')
        if valid_loss <= min(valid_hist):
            save_ckpt(work_dir, model, mode='best')

        print(f'epoch {epoch+1}: train_loss={train_loss:>3f}, '
              f'valid_loss={valid_loss:>3f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-type', required=True,
                        help='model type to use')
    parser.add_argument('-d', '--data-dir', required=True,
                        help='prepared dir with preprocessed data')
    parser.add_argument('-w', '--work-dir', type=str, default='./out',
                        help='working dir to output')
    parser.add_argument('-n', '--num-epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size of mini-batch')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate of training')
    parser.add_argument('--checkpoint', action='store_true',
                        help='whether use checkpoint in working dir')
    args = parser.parse_args()

    src_vocab, tgt_vocab = load_vocab(args.data_dir)
    train_set, valid_set = create_dataset(args.data_dir,
                                          vocab=(src_vocab, tgt_vocab),
                                          split=('train', 'valid'))
    model = create_model(model_type=args.model_type,
                         enc_vocab=len(src_vocab),
                         dec_vocab=len(tgt_vocab))

    if args.checkpoint:
        load_ckpt(model, args.work_dir)

    train(model, train_set, valid_set, src_vocab, tgt_vocab,
          work_dir=args.work_dir,
          num_epochs=args.num_epochs,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate)
