import torch
from dnaset.util import seq_to_onehot
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
'''
pred = Enformer.from_hparams(
	dim = 1536,
	depth = 11,
	heads = 8,
	output_heads = dict(human = 5313),
	target_length = 896,
	use_checkpointing=True

)
'''
pred = Enformer.from_pretrained('EleutherAI/enformer-official-rough')

print("model built")
model_pth = "/projectnb/aclab/vraiti/models/enformer/save/params"
if len(sys.argv)==2:
	prev_its = int(sys.argv[1])
	model_pth+=str(prev_its)+".pt"
elif len(sys.argv)>2:
	print("usage: train.py [specific checkpoint]")
	exit()
else:
	model_pth+=".pt"



#pred.load_state_dict(torch.load(model_pth))
data = BasenjiDataset(
	organism= "human",
	split = "train",
	seq_length=114688
)

optimizer = Adam(pred.parameters())
batch_size = 1
dl = DataLoader(data,batch_size=batch_size,shuffle=False)
last_time = time.monotonic()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#pred = torch.compile(pred,mode="reduce-overhead")
pred = pred.to(device)

for it, seq_dict in enumerate(dl):
    bpstr = seq_dict['sequence'] #base pairs
    x = str_to_one_hot(bpstr)
    x = x.to(device)
    y = seq_dict['target'].to(device)
    ret = pred(
		x=x,
		target=y,
		return_corr_coef=True,
		head='human'
	)
    loss = ret['loss']
    coef = ret['corr_coef']

    loss.backward()
    optimizer.step()

    msg=f"loss: {loss}, corr coef: {coef}, it/second: {batch_size/(time.monotonic()-last_time)}\n:"
    print(msg)
    write("train.py.out",msg)

    last_time = time.monotonic()
    if it%20==0:
        torch.save(pred.state_dict(),f"/projectnb/aclab/vraiti/models/enformer/save/params{it}.pt")
torch.save(pred.state_dict(),f"/projectnb/aclab/vraiti/models/enformer/save/params.pt")
