import torch


def beam_initial(out, beam):
    # out: (batch, 1, dec_vocab)
    probs, indics = out[:,-1,:].topk(beam, dim=-1)
    # (batch, beam)
    return indics.unsqueeze(-1), torch.log(probs)

def beam_search(out, preds, scores, beam):
    bs = scores.shape[0]
    # out: (batch*beam, 1, dec_vocab)
    # preds: (batch, beam, t), scores: (batch, beam)
    probs, indics = out[:,-1,:].topk(beam, dim=-1)
    # probs, indics: (batch*beam, beam)
    probs = probs.view(bs, beam, beam)
    indics = indics.view(bs, beam, beam)
    # (batch, beam, beam)
    scores = scores.unsqueeze(-1) + torch.log(probs)
    # (batch, beam, beam)
    scores, b_indics = scores.view(bs, -1).topk(beam, dim=-1)
    # scores: (batch, beam), (batch, beam)
    b_row, b_col = b_indics.div(beam).long(), b_indics.fmod(beam).long()
    batch = torch.arange(bs).unsqueeze(1).repeat(1, beam)
    index = indics[batch.view(-1), b_row.view(-1), b_col.view(-1)].long()
    # index: (batch*beam,)
    preds = torch.cat([preds, index.view(bs, beam, 1)], dim=-1)
    return preds, scores
