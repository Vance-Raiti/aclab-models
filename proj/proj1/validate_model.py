import torch
import bigwig_dataset
from pyfaidx import Fasta
import locs
from pybedtools import BedTool
import pyBigWig
import math
import torch.backends.cudnn as cudnn
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pred = model.Model(n_blocks=8,n_steps=2,in_channels=4)
pred.load_state_dict(torch.load(locs.MODEL_STATE_DICT,map_location=device))


pred = pred.to(device)
if device == 'cuda':
    pred = torch.nn.DataParallel(pred)
    cudnn.benchmark = True
print(f"device is {device}")

loss_fn = torch.nn.MSELoss()

data = bigwig_dataset.BigWigDataset(
    bigwig_files = locs.BIGWIG_FILE,
    reference_fasta_file=locs.HUMAN_FA,
    input_bed_file=locs.TILED_BED
)

valid_inds_range = range(math.floor(len(data)*0.8),math.floor(len(data)*0.9))
valid_loss = 0
for idx in valid_inds_range:
    try:
        data_i = data[idx]
    except RuntimeError:
        print(f"{idx} is out of bounds")
        continue
    seq = torch.Tensor(data_i['sequence'])
    seq = seq.to(device)
    target = torch.Tensor(data_i['values'][0])
    target = target.to(device)

    
    x = model.encode_sequence(seq)
    if x is None:
        print(f"skipped sequence {idx}. Most likely contains nonzero count of 'N'")
        continue
    with torch.no_grad():
        valid_loss += loss_fn(target,pred(x)).item()/math.floor(len(data)*0.2)
    
    

print(f'Validation loss: {valid_loss}')