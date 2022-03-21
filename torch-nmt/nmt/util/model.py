import torch

def load_model(infile, model, optimizer):
    print(f'Loading model from {infile}...')
    checkpoint = torch.load(infile)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def save_model(outfile, model, optimizer, src_vocab, tgt_vocab):
    print(f'Saving model to {outfile}...')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, outfile)
