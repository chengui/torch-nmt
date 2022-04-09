import os
import torch
from nmt.model.seq2seq_gru import RNNSeq2Seq
from nmt.model.seq2seq_luong import LuongSeq2Seq
from nmt.model.seq2seq_bahdanau import BahdanauSeq2Seq
from nmt.model.seq2seq_transformer import TransformerSeq2Seq

MODELS = {
    'rnn':         RNNSeq2Seq,
    'luong':       LuongSeq2Seq,
    'bahdanau':    BahdanauSeq2Seq,
    'transformer': TransformerSeq2Seq,
}

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    else:
        for name, param in m.named_parameters():
            if 'weight' in name and param.dim() > 1:
                torch.nn.init.xavier_uniform_(param.data)

def create_model(model_type, enc_vocab, dec_vocab, **kw):
    seq2seq = MODELS[model_type](enc_vocab, dec_vocab)
    seq2seq.apply(init_weights)
    return seq2seq

def load_ckpt(model, work_dir, mode='last'):
    model_dir = os.path.join(work_dir, 'model')
    if not os.path.exists(model_dir):
        raise OSError(f'model dir not exits: {work_dir}')
    model_file = os.path.join(model_dir, f'checkpoint-{mode}.pt')
    if not os.path.exists(model_file):
        raise OSError(f'checkpoint not exists: {work_dir}')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

def save_ckpt(work_dir, model, mode='last'):
    model_dir = os.path.join(work_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, f'checkpoint-{mode}.pt')
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, model_file)
