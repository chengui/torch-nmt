import os
import torch
from nmt.model.seq2seq_rnn import RNNSeq2Seq
from nmt.model.seq2seq_luong import LuongSeq2Seq
from nmt.model.seq2seq_bahdanau import BahdanauSeq2Seq
from nmt.model.seq2seq_transformer import TransformerSeq2Seq

MODELS = {
    'rnn':         RNNSeq2Seq,
    'luong':       LuongSeq2Seq,
    'bahdanau':    BahdanauSeq2Seq,
    'transformer': TransformerSeq2Seq,
}

def create_model(enc_vocab, dec_vocab, **kw):
    model_type = kw.get('type', None)
    model_params = kw.get('params', None)
    if not model_type or not model_params:
        raise KeyError('invalid model configure')

    Seq2Seq = MODELS.get(model_type, None)
    if not Seq2Seq:
        raise KeyError('invalid model type')
    params = model_params.get(model_type, None)
    if not params:
        raise KeyError('invalid model params')

    seq2seq = Seq2Seq(enc_vocab, dec_vocab, **params)
    seq2seq.apply(init_weights)
    return seq2seq

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    else:
        for name, param in m.named_parameters():
            if 'weight' in name and param.dim() > 1:
                torch.nn.init.xavier_uniform_(param.data)

def load_ckpt(work_dir, model, optimizer=None, mode='last'):
    model_dir = os.path.join(work_dir, 'model')
    if not os.path.exists(model_dir):
        raise OSError(f'model dir not exits: {work_dir}')
    model_file = os.path.join(model_dir, f'checkpoint-{mode}.pt')
    if not os.path.exists(model_file):
        raise OSError(f'checkpoint not exists: {work_dir}')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim'])

def save_ckpt(work_dir, model, optimizer, mode='last'):
    model_dir = os.path.join(work_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, f'checkpoint-{mode}.pt')
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
    }
    torch.save(checkpoint, model_file)
