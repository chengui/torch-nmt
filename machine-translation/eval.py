from torch import nn
from torch.utils.data import DataLoader
from models import create_model
from datasets import create_dataset
from datasets.utils import load_vocab
from utils import load_model, bleu_score


def evaluate(model, dataset, vocab, batch_size):
    test_iter = DataLoader(dataset=dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    test_loss, test_bleu = 0, 0
    for idx, (src, tgt) in enumerate(test_iter):
        pred = model(src, tgt)
        # pred: [seq_len, batch_size, vocab_size]
        pred = pred.permute(1, 2, 0)
        bleu = bleu_score(vocab[pred], vocab[tgt])
        loss = criterion(pred[:, :, 1:], tgt[:, 1:])
        test_loss += loss.item()
        test_bleu += bleu
    test_loss /= len(test_iter)
    print(f'Test Error: loss={test_loss:>3f}, bleu={test_bleu:>3f}')
    return test_loss, test_bleu


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
    parser.add_argument('-t', '--trained', default='classify.pt',
                        help='trained model')
    args = parser.parse_args()

    model_options = {
        'embed_size': 32,
        'num_hiddens': 32,
        'num_layers': 2,
        'dropout': 0.1
    }

    dataset = create_dataset(args.dataset, split='test',
                             root_dir=args.root_dir)
    src_vocab, tgt_vocab = load_vocab()
    vocab_size = (len(src_vocab), len(tgt_vocab))
    model = create_model(args.model,
                         vocab_size=vocab_size,
                         **model_options)

    load_model(args.trained, model)
    evaluate(model, dataset, args.vocab, args.batch_size)
