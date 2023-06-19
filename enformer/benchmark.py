import torch
from enformer_pytorch import Enformer
from enformer_pytorch.config_enformer import EnformerConfig
from basenji import BasenjiDataset
from filewriter import write
from olltrainer.trainer import Trainer, Writer, TrainerConfig
from torch.optim import Adam
import time
import wandb

BATCH_SIZE = 1
log_pth = "logs/benchmark.out"

class EnformerWriter(Writer):
	def __init__(self):
		self.running_loss = 0.0
		self.it = 0
		self.running_r = 0.0 #correlation coefficient
		self.last_time = time.monotonic()
		self.last_it = 0

	def record(self,model_output):
		self.running_loss += model_output['loss'].item()
		self.it += 1
		self.running_r += model_output['correlation_coefficient'].item()
	
	def write(self):
		nits = self.it-self.last_it
		msg = f"{self.it} -- avg loss: {self.running_loss/nits} avg r: {self.running_r/nits} it/sec: {nits*BATCH_SIZE/(time.monotonic()-self.last_time)}"
		print(msg)
		write(msg,log_pth)
		
		self.running_loss = 0.0
		self.running_r = 0.0
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



data = BasenjiDataset(
	organism= "human",
	split = "valid",
	seq_length=114688
)


train_conf = TrainerConfig(
	write_interval = 20,
    max_write_interval = 100,
	save_interval = 1000,
	batch_size = BATCH_SIZE,
	writer = EnformerWriter(),
    forward_targets = True,
    use_model_loss = True,
    epochs=5,
    use_eval_test = False
)

train_obj = Trainer(
	model = pred,
	config = train_conf,	
	test_data = data
)

train_obj.test()
