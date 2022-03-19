import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import create_dataset, split_dataset
from models import create_model
from utils import plot_history, bleu_score
from utils import load_model, save_model


def valid_epoch(model, data_iter, vocab):
    with torch.no_grad():
        valid_bleu = 0.0
        for idx, (src, tgt) in enumerate(data_iter):
            pred = model(src, tgt)
            # pred: [seq_len, batch_size, vocab_size]
            # tgt: [batch_size, seq_len]
            pred = pred.argmax(2).permute(1, 0)
            # pred: [batch_size, seq_len]
            pred = [vocab.token(ids) for ids in pred]
            tgt = [vocab.token(ids) for ids in tgt]
            bleu = bleu_score(pred, tgt)
            valid_bleu += bleu
        valid_bleu /= len(data_iter)
        return valid_bleu

def train_epoch(model, data_iter, optimizer, criterion):
    train_loss = 0
    for idx, (src, tgt) in enumerate(data_iter):
        optimizer.zero_grad()

        pred = model(src, tgt)
        # pred: [seq_len, batch_size, vocab_size]
        pred = pred.permute(1, 2, 0)

        loss = criterion(pred[:, :, 1:], tgt[:, 1:])
        loss.backward()
        #clip_grad
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(data_iter)
    return train_loss

def train(model, dataset, num_epochs, batch_size, lr, pretrain, output):
    src_vocab, tgt_vocab =  dataset.src_vocab, dataset.tgt_vocab
    trainset, validset = split_dataset(dataset, lengths=[0.8, 0.2])

    train_iter = DataLoader(dataset=trainset, batch_size=batch_size)
    valid_iter = DataLoader(dataset=validset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if pretrain:
        load_model(pretrain, model, optimizer)
    else:
        model.apply(init_weights)

    loss_lst, bleu_lst = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_iter, optimizer, criterion)
        valid_bleu = valid_epoch(model, valid_iter, tgt_vocab)

        loss_lst.append(train_loss)
        bleu_lst.append(valid_bleu)
        print(f'epoch {epoch + 1}: loss={train_loss:>3f}, bleu={valid_bleu:>3f}')

    save_model(output, model, optimizer, src_vocab, tgt_vocab)
    plot_history(output,
                 legend=['train loss', 'valid bleu'],
                 plots=[loss_lst, bleu_lst])

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

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
    parser.add_argument('-t', '--pretrain', default='classify.pt',
                         help='pretrain model')
    parser.add_argument('-o', '--output', default='classify.pt',
                        help='output model')
    args = parser.parse_args()

    dataset = create_dataset(dataset=args.dataset,
                             root_dir=args.root_dir)
    vocab_size = (len(dataset.src_vocab), len(dataset.tgt_vocab))
    model = create_model(model=args.model,
                         vocab_size=vocab_size)

    train(model, dataset, args.epochs, args.batch_size, args.lr, args.pretrain, args.output)
