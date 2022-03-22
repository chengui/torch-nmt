import os
import torch
from .vocab import Vocab


def load_checkpoint(ptfile, model, optimizer=None):
    if not os.path.exists(ptfile):
        return
    print(f'Loading checkpoint from {ptfile}...')
    checkpoint = torch.load(ptfile)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

def save_checkpoint(ptfile, model, optimizer=None):
    print(f'Saving checkpoint to {ptfile}...')
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint.update({
            'optimizer': optimizer.state_dict(),
        })
    torch.save(checkpoint, ptfile)

def load_vocab(indir, src_file='src_vocab.txt', tgt_file='tgt_vocab.txt'):
    src_file = os.path.join(indir, src_file)
    if os.path.exists(src_file):
        print(f'Loading src_vocab from {src_file}...')
        src_vocab = Vocab.from_file(src_file)
    else:
        print(f'[WARN] {src_file} not exists')
    tgt_file = os.path.join(indir, tgt_file)
    if os.path.exists(tgt_file):
        print(f'Loading tgt_vocab from {tgt_file}...')
        tgt_vocab = Vocab.from_file(tgt_file)
    else:
        print(f'[WARN] {tgt_file} not exists')
    return src_vocab, tgt_vocab

def save_vocab(src_vocab, tgt_vocab, outdir='.'):
    src_file = src_vocab.name if src_vocab.name else 'src_vocab'
    src_file = os.path.join(outdir, src_file + '.txt')
    if not os.path.exists(src_file):
        print(f'Saving src_vocab to {src_file}...')
        src_vocab.to_file(src_file)
    tgt_file = tgt_vocab.name if tgt_vocab.name else 'tgt_vocab'
    tgt_file = os.path.join(outdir, tgt_file + '.txt')
    if not os.path.exists(tgt_file):
        print(f'Saving tgt_vocab to {tgt_file}...')
        tgt_vocab.to_file(tgt_file)
