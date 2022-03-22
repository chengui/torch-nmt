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

def load_vocab(src_file, tgt_file, indir='.'):
    print(f'Loading vocab to {indir}...')
    src_vocab = Vocab.from_file(os.path.join(indir, src_file))
    tgt_vocab = Vocab.from_file(os.path.join(indir, tgt_file))
    return src_vocab, tgt_vocab

def save_vocab(src_vocab, tgt_vocab, outdir='.'):
    print(f'Saving vocab to {outdir}...')
    src_file = src_vocab.name if src_vocab.name else 'src_vocab'
    src_vocab.to_file(os.path.join(outdir, src_file + '.txt'))
    tgt_file = tgt_vocab.name if tgt_vocab.name else 'tgt_vocab'
    tgt_vocab.to_file(os.path.join(outdir, tgt_file + '.txt'))
