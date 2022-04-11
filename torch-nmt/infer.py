import os
import torch
from nmt.vocab import load_vocab
from nmt.dataset.data import numerical
from nmt.model import (
    create_model,
    load_ckpt,
)


def batch_toindex(tokens, vocab):
    if not isinstance(tokens, (tuple, list)):
        tokens = [tokens]
    tokens = [vocab[token] for token in tokens]
    return tokens

def batch_totoken(indics, vocab, unsqueeze=False, strip_eos=True):
    if isinstance(indics, torch.Tensor):
        indics = indics.tolist()
    filtered = lambda i: i not in (vocab.PAD_IDX, vocab.SOS_IDX)
    batch = []
    for sent in indics:
        sent = list(filter(filtered, sent))
        if vocab.EOS_IDX in sent:
            i = sent.index(vocab.EOS_IDX)
            sent = sent[:i] if strip_eos else sent[:i+1]
        if unsqueeze:
            batch.append([vocab.token(sent)])
        else:
            batch.append(vocab.token(sent))
    return batch

def predict(model, sents, src_vocab, tgt_vocab, pred_file=None, maxlen=20):
    sos_idx, pad_idx = src_vocab.SOS_IDX, src_vocab.PAD_IDX
    sos_indic, sos_len_indic = [sos_idx] + [pad_idx] * (maxlen-1), [1]
    model.eval()
    pred_seq = []
    for _, sent in enumerate(sents):
        src, src_len = numerical([sent], src_vocab)
        sos = torch.LongTensor(sos_indic).unsqueeze(0).repeat(src.shape[0], 1)
        sos_len = torch.LongTensor(sos_len_indic).repeat(src.shape[0])
        pred = model(src, src_len, sos, sos_len, teacher_ratio=0)
        pred_seq.extend(batch_totoken(pred.argmax(2), tgt_vocab))
    with open(pred_file, 'w', encoding='utf-8') as wf:
        for tokens in pred_seq:
            wf.write(' '.join(tokens) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-type', required=True,
                        help='model type to use')
    parser.add_argument('-w', '--work-dir', required=True,
                        help='working dir to output')
    parser.add_argument('-f', '--source-file', required=True,
                        help='source file with preprocessed data')
    parser.add_argument('-l', '--max-length', type=int, default=10,
                        help='maxium length to predict')
    args = parser.parse_args()

    src_vocab, tgt_vocab = load_vocab(args.work_dir)
    model = create_model(model_type=args.model_type,
                         enc_vocab=len(src_vocab),
                         dec_vocab=len(tgt_vocab))
    load_ckpt(model, args.work_dir, mode='best')

    with open(args.source_file, 'r') as f:
        sents = [l.strip().split(' ') for l in f]
    out_dir = os.path.join(args.work_dir, 'out')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pred_name = os.path.basename(args.source_file) + '.pred'
    pred_file = os.path.join(out_dir, pred_name)
    predict(model, sents, src_vocab, tgt_vocab, pred_file, args.max_length)
