import torch
from nmt.util import get_device
from nmt.workdir import WorkDir
from nmt.config import Config
from nmt.vocab import (
    load_vocab,
    batch_totoken,
)
from nmt.dataset.data import (
    tolist,
    numerical,
    init_target,
)
from nmt.model import (
    create_model,
    load_ckpt,
)


@torch.no_grad()
def predict(model, sents, src_vocab, tgt_vocab, device=None, beam=None,
            pred_file=None, maxlen=10):
    model.eval()
    pred_seq = []
    eos_idx = tgt_vocab.EOS_IDX
    for _, sent in enumerate(sents):
        src, src_len = numerical([sent], src_vocab, maxlen, device)
        sos, sos_len = init_target(src.shape[0], src_vocab, maxlen, device)
        if not beam:
            pred, lens = model.predict(src, src_len, sos, sos_len,
                                       eos_idx=eos_idx, maxlen=maxlen)
            pred_list = tolist(pred.unsqueeze(1), lens.unsqueeze(1))
        else:
            pred, lens = model.beam_predict(src, src_len, sos, sos_len,
                                            beam=beam, eos_idx=eos_idx,
                                            maxlen=maxlen)
            pred_list = tolist(pred, lens)
        pred_seq.extend(batch_totoken(pred_list, tgt_vocab))
    with open(pred_file, 'w', encoding='utf-8') as wf:
        for (sent, preds) in zip(sents, pred_seq):
            for pred in preds:
                wf.write(' '.join(sent) + '\t' + ' '.join(pred) + '\n')
        print(f'Predicted result stores at {pred_file}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='configure file for model')
    parser.add_argument('-w', '--work-dir', required=True,
                        help='working dir to output')
    parser.add_argument('-m', '--model-type', default=None,
                        help='model type to use')
    parser.add_argument('-f', '--source-file', required=True,
                        help='source file with preprocessed data')
    parser.add_argument('-l', '--max-length', type=int, default=10,
                        help='maxium length to predict')
    parser.add_argument('-b', '--beam-size', type=int, default=None,
                        help='beam size used in beam search')
    parser.add_argument('--onlycpu', action='store_true',
                        help='whether only work on cpu')
    args = parser.parse_args()

    wdir = WorkDir(args.work_dir)
    conf = Config.load_config(args.config)
    if args.model_type:
        conf.model.update({'type': args.model_type})

    src_vocab, tgt_vocab = load_vocab(wdir.vocab)
    device = get_device(args.onlycpu)
    model = create_model(enc_vocab=len(src_vocab),
                         dec_vocab=len(tgt_vocab),
                         **conf.model)
    model = model.to(device)

    model_dir = wdir.model.sub(conf.model.type)
    load_ckpt(model_dir, model, None, mode='best')

    with open(wdir.test.rfile(args.source_file), 'r') as f:
        sents = [l.strip().split(' ') for l in f]
    pred_file = wdir.out.file(f'{args.source_file}.pred')
    predict(model, sents, src_vocab, tgt_vocab,
            device=device,
            beam=args.beam_size,
            pred_file=pred_file,
            maxlen=args.max_length)
