import torch
from nmt.processor.base import BaseProcessor
from nmt.model import create_model
from nmt.util.fs_util import load_checkpoint, load_vocab


def predict_one(encoder, decoder, enc_seq, dec_seq, maxlen, eos_idx):
    if not isinstance(enc_seq, torch.Tensor):
        enc_seq = torch.tensor(enc_seq)
    if not isinstance(dec_seq, torch.Tensor):
        dec_seq = torch.tensor(dec_seq)
    enc_X = enc_seq.unsqueeze(0)
    dec_X = dec_seq.unsqueeze(0)
    _, state = encoder(enc_X)
    outputs = []
    X = dec_X[:, 0]
    for _ in range(maxlen):
        output, state = decoder(X, state)
        X = output.argmax(1)
        pred = X.squeeze(0).item()
        if pred == eos_idx:
            break
        outputs.append(pred)
    return outputs

def predict(model, processor, src_vocab, tgt_vocab, src_texts, maxlen=10):
    model.eval()
    if isinstance(processor, (list, tuple)):
        src_processor, _ = processor
    else:
        src_processor, _ = processor, processor
    src_data, _ = src_processor(src_texts, src_vocab)
    encoder, decoder = model.encoder, model.decoder
    eos_idx = tgt_vocab.EOS_IDX
    for (sent, src) in zip(src_texts, src_data):
        tgt = [tgt_vocab.EOS_IDX]
        pred = predict_one(encoder, decoder, src, tgt, maxlen, eos_idx)
        pred = ' '.join(tgt_vocab.token(pred))
        print(f'> {sent}')
        print(f'< {pred}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='rnn',
                        help='model to use')
    parser.add_argument('-s', '--sents', action='append',
                        help='sentences')
    parser.add_argument('-t', '--trained', default='./output',
                        help='trained model')
    args = parser.parse_args()

    src_vocab, tgt_vocab = load_vocab(args.trained)
    vocab_size = (len(src_vocab), len(tgt_vocab))
    model = create_model(args.model, vocab_size=vocab_size)
    load_checkpoint(f'{args.trained}/model.pt', model)
    processor = BaseProcessor()

    predict(model, processor, src_vocab, tgt_vocab, args.sents)
