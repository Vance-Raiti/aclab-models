import torch
import bigwig_dataset
from pyfaidx import Fasta
import locs
from pybedtools import BedTool
import pyBigWig
import math
import torch.backends.cudnn as cudnn

data = bigwig_dataset.BigWigDataset(
    bigwig_files = locs.BIGWIG_FILE,
    reference_fasta_file=locs.HUMAN_FA,
    input_bed_file=locs.TILED_BED
)

s = 0
for d in data:
    seq = torch.Tensor(d["sequence"])
    na = seq!=65
    nc = seq!=67
    ng = seq!=71
    nt = seq!=84
    tand = torch.logical_and
    key_errors = tand(tand(na,nc),tand(ng,nt))
    s += torch.sum(key_errors)
    print(s)
print(s)

