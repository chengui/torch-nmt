import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from nmt.processor.base import BaseProcessor
from nmt.model import create_model
from nmt.dataset import (
    create_dataset,
    split_dataset
)
from nmt.util.fs_util import (
    load_checkpoint,
    save_checkpoint,
    save_vocab
)


def valid_epoch(model, data_iter, criterion):
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for idx, (src, tgt) in enumerate(data_iter):
            pred = model(src, tgt)
            # pred: (batch_size, seq, vocab_size)
            # tgt: (batch_size, seq)
            pred = pred.permute(0, 2, 1)
            loss = criterion(pred[:, :, 1:], tgt[:, 1:])
            valid_loss += loss
        valid_loss /= len(data_iter)
        return valid_loss

def train_epoch(model, data_iter, criterion, optimizer):
    model.train()
    train_loss = 0
    for idx, (src, tgt) in enumerate(data_iter):
        optimizer.zero_grad()
        pred = model(src, tgt)
        # pred: (batch_size, seq, vocab_size)
        # tgt: (batch_size, seq)
        pred = pred.permute(0, 2, 1)
        loss = criterion(pred[:, :, 1:], tgt[:, 1:])
        loss.backward()
        #clip_grad
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(data_iter)
    return train_loss

def train(model, dataset, num_epochs, batch_size, lr, pretrain, outdir):
    src_vocab, tgt_vocab =  dataset.src_vocab, dataset.tgt_vocab
    trainset, validset = split_dataset(dataset, ratios=[0.8, 0.2])

    train_iter = DataLoader(dataset=trainset, batch_size=batch_size)
    valid_iter = DataLoader(dataset=validset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if pretrain:
        load_checkpoint(pretrain, model, optimizer)

    train_hist, valid_hist = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_iter, criterion, optimizer)
        valid_loss = valid_epoch(model, valid_iter, criterion)

        train_hist.append(train_loss)
        valid_hist.append(valid_loss)
        print(f'epoch {epoch + 1}: train_loss={train_loss:>3f}, '
              f'valid_loss={valid_loss:>3f}')

        if (epoch + 1) % 10 == 0:
            save_checkpoint(f'{outdir}/model.pt', model, optimizer)
            save_vocab(src_vocab, tgt_vocab, outdir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='tatoeba',
                        help='dataset to use')
    parser.add_argument('-m', '--model', default='rnn',
                        help='model to use')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='batch size of dataloader')
    parser.add_argument('-l', '--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('-n', '--epochs', type=int, default=10,
                        help='number of train epochs')
    parser.add_argument('-r', '--root-dir', default='.',
                         help='root dir of dataset')
    parser.add_argument('-t', '--pretrain', default='nmt.pt',
                         help='pretrain model')
    parser.add_argument('-o', '--outdir', default='./output',
                        help='output dir')
    args = parser.parse_args()

    dataset = create_dataset(dataset=args.dataset,
                             root_dir=args.root_dir,
                             processor=BaseProcessor())
    vocab_size = (len(dataset.src_vocab), len(dataset.tgt_vocab))
    model = create_model(model=args.model, vocab_size=vocab_size)

    train(model, dataset, args.epochs, args.batch_size, args.lr, args.pretrain, args.outdir)
