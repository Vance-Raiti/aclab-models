import torch

def encode(seq):

    na = seq!=65
    nc = seq!=67
    ng = seq!=71
    nt = seq!=84
    tand = torch.logical_and
    if torch.any(tand(tand(na,nc),tand(ng,nt))):
        return None
    seq = (seq==65)*0+(seq==67)*1+(seq==84)*2+(seq==71)*3
    out = seq.type(torch.int64)
    return out