import torch
import time
from enformer_pytorch import Enformer, str_to_one_hot
from enformer_pytorch.config_enformer import EnformerConfig
import sys
sys.path.append("/projectnb/aclab/datasets/dataloaders/dataloaders/basenji")
from basenji import BasenjiDataset
from torch.utils.data import DataLoader
from filewriter import write
from torch.optim import Adam

print("building model")
device = "cuda" if torch.cuda.is_available() else "cpu"

pred = Enformer.from_hparams(
	dim = 1536,
	depth = 11,
	heads = 8,
	output_heads = dict(human = 5313),
	target_length = 896,
	use_checkpointing=True
	
)
pred = Enformer.from_pretrained('EleutherAI/enformer-official-rough',use_checkpointing=True)
pred = pred.to(device)
pred.eval()
print("model built")
#pred.load_state_dict(torch.load("/projectnb/aclab/vraiti/models/enformer/params200.pt"))
print("model loaded")
data = BasenjiDataset(
	organism= "human",
	split = "train",
	seq_length=114688
)

batch_size = 1
dl = DataLoader(data,batch_size=batch_size,shuffle=False)
last_time = time.monotonic()
running_loss = 0.0
running_coef = 0.0
with torch.autocast(device):
	for it, seq_dict in enumerate(dl):
		bpstr = seq_dict['sequence'] #base pairs
		x = str_to_one_hot(bpstr)
		y = seq_dict['target']
		x = x.to(device)
		y = y.to(device)
		ret = pred(
			x=x,
			target=y,
			return_corr_coef=True,
			head='human'
		)
		loss = ret['loss'].item()
		coef = ret['corr_coef'].item()
		running_coef += coef
		running_loss += loss
		if it%20==0:
			msg = f"loss: {loss}, corr coef: {coef}, it/second: {batch_size/(time.monotonic()-last_time)}"
			print(msg)
			write("test.py.out",msg+"\n")
			last_time = time.monotonic()
msg = "avg corr coef: {running_loss/len(data)}"
print(msg)
write("test.py.out",msg+"\n")
