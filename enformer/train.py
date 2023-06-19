import torch
from enformer_pytorch import Enformer
from enformer_pytorch.config_enformer import EnformerConfig
from basenji import BasenjiDataset
from filewriter import write
from olltrainer.trainer import Trainer, Writer, TrainerConfig
import torch.optim as optim
import time
import wandb
import os


BATCH_SIZE = 1
EPOCHS = 5
MAX_LR = 1e-2
log_pth = "logs/write.out"
os.remove(log_pth)

class EnformerWriter(Writer):
	def __init__(self):
		self.running_loss = 0.0
		self.it = 0
		self.running_r = 0.0 #correlation coefficient
		self.last_time = time.monotonic()
		self.last_it = 0
		self.running_out = 0.0

	def record(self,model_output):
		self.running_loss += model_output['loss'].item()
		self.it += 1
		self.running_r += model_output['correlation_coefficient'].item()
		self.running_out += torch.mean(model_output['loss']).item()	

	def write(self):
		nits = self.it-self.last_it
		msg = f"{self.it} -- avg loss: {self.running_loss/nits} avg r: {self.running_r/nits} avg output: {self.running_out/nits} it/sec: {nits*BATCH_SIZE/(time.monotonic()-self.last_time)}\n"
		print(msg)
		write(msg,log_pth)
		
		self.running_loss = 0.0
		self.running_r = 0.0
		self.running_out = 0.0
		self.last_time = time.monotonic()
		self.last_it = self.it


pred = Enformer.from_hparams(
	dim = 1536,
	depth = 11,
	heads = 8,
	output_heads = dict(human = 5313),
	target_length = 896,
	use_checkpointing = True
)

optimizer = optim.Adam(
    pred.parameters(),
    lr=5e-4,
)

lr_scheduler= optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda n: min(n*MAX_LR/5000,MAX_LR)
)

data = BasenjiDataset(
	organism= "human",
	split = "train",
	seq_length=114688
)
        

train_conf = TrainerConfig(
	ckpt_pth = "save/enformer",
	write_interval = 100,
    save_interval = 1000,
	batch_size = BATCH_SIZE,
	writer = EnformerWriter(),
    forward_targets = True,
    use_model_loss = True,
    use_autocast = False,
    epochs=EPOCHS,
    max_grad_norm=0.2
)

train_obj = Trainer(
	model = pred,
	config = train_conf,
	optimizer = optimizer,	
	train_data = data
    lr_scheduler = lr_scheduler
)

train_obj.train()
