import torch
import bigwig_dataset
from pyfaidx import Fasta
import locs
from pybedtools import BedTool
import pyBigWig
import math
import torch.backends.cudnn as cudnn
import model


print("starting!")

data = bigwig_dataset.BigWigDataset(
    bigwig_files = locs.BIGWIG_FILE,
    reference_fasta_file=locs.HUMAN_FA,
    input_bed_file=locs.TRAIN_BED
)

print("dataloader built")
print(f"dataloader is size {len(data)}")
pred = model.Model(n_blocks=8,n_steps=2,in_channels=4)

print("model built")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pred = pred.to(device)
if device == 'cuda':
    pred = torch.nn.DataParallel(pred)
    cudnn.benchmark = True
print(f"device is {device}")
optimizer = torch.optim.SGD(pred.parameters(), lr=0.1,momentum=0.9)
optimizer.zero_grad()





train_inds_range = range(0,math.floor(len(data)*0.8))




#train loop
print("training...")
for epoch in range(2):
    epoch_loss = 0

    for idx in train_inds_range:
        optimizer.zero_grad()
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
        y_hat = pred(x)
        loss = loss_fn(target,y_hat)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            epoch_loss += loss/math.floor(len(data)*0.8)
        if idx%100==0:
            print(f'{idx}->{loss}')
    print(f'{epoch}: {epoch_loss}')

torch.save(pred.state_dict(), "/projectnb/aclab/vraiti/proj/model.pth")