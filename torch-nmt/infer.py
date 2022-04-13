import os
from nmt.util import get_device
from nmt.vocab import (
    load_vocab,
    batch_totoken,
)
from nmt.dataset.data import (
    numerical,
    init_target,
)
from nmt.model import (
    create_model,
    load_ckpt,
)


def predict(model, sents, src_vocab, tgt_vocab, device=None, pred_file=None,
            maxlen=10):
    model.eval()
    pred_seq = []
    for _, sent in enumerate(sents):
        src, src_len = numerical([sent], src_vocab, maxlen, device)
        sos, sos_len = init_target(src.shape[0], src_vocab, maxlen, device)
        out = model(src, src_len, sos, sos_len, teacher_ratio=0)
        pred = out.argmax(2).tolist()
        pred_seq.extend(batch_totoken(pred, tgt_vocab, strip_eos=True))
    with open(pred_file, 'w', encoding='utf-8') as wf:
        for (sent, pred) in zip(sents, pred_seq):
            wf.write(' '.join(sent) + '\t' + ' '.join(pred) + '\n')


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
    parser.add_argument('--cpu-only', action='store_true',
                        help='whether work on cpu only')
    args = parser.parse_args()

    src_vocab, tgt_vocab = load_vocab(args.work_dir)
    device = get_device(args.cpu_only)
    model = create_model(model_type=args.model_type,
                         enc_vocab=len(src_vocab),
                         dec_vocab=len(tgt_vocab))
    model = model.to(device)
    load_ckpt(args.work_dir, model, None, mode='best')

    with open(args.source_file, 'r') as f:
        sents = [l.strip().split(' ') for l in f]
    out_dir = os.path.join(args.work_dir, 'out')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pred_name = os.path.basename(args.source_file) + '.pred'
    pred_file = os.path.join(out_dir, pred_name)
    predict(model, sents, src_vocab, tgt_vocab,
            device=device,
            pred_file=pred_file,
            maxlen=args.max_length)
