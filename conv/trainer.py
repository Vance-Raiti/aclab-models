"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""


assert __name__ != "__main__" , "'trainer.py' should only be imported"
   

import math
import logging
import time

from tqdm import tqdm
import numpy as np
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from encoder import encode
import sys

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    ckpt_interval = None
    ckpts_to_keep = 5
    num_workers = 0 # for DataLoader
    optimizer = 'adamw'
    logdir ='./runs/'
    load_model = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.ckpts = []

        if config.ckpt_path.endswith('.pt'):
            config.ckpt_path = config.ckpt_path[:-3]
        if config.ckpt_path.endswith('.pth'):
            config.ckpt_path = config.ckpt_path[:-4]
        if config.ckpt_path.endswith('/'):
            config.ckpt_path = config.ckpt_path+'checkpoint'
            

        self.config = config


        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:'+str(torch.cuda.current_device())
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, it=None):
        ckpt_path = self.config.ckpt_path
        if ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            if it is None:
                ckpt_path = ckpt_path + '.pt'
            else:
                ckpt_path = ckpt_path + '-' + str(it)+'.pt'
            logger.info("saving %s", ckpt_path)
            # torch.save(ckpt_model.state_dict(), ckpt_path)
            # torch.save({
            #     "state_dict": ckpt_model.state_dict(),
            #     "config": ckpt_model.config()
            #     }, ckpt_path)
            ckpt_model.save_to_checkpoint(ckpt_path)
            self.ckpts.append(ckpt_path)

            if len(self.ckpts) > self.config.ckpts_to_keep:
                os.remove(self.ckpts[0])
                self.ckpts.remove(self.ckpts[0])

    def train(self, progress_bar=True):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        # params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        # params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        # optim_groups = [
        #     {"params": params_decay, "weight_decay": config.weight_decay},
        #     {"params": params_nodecay, "weight_decay": 0.0},
        # ]
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.Adam(raw_model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        # if config.optimizer == 'adamw':
        #     optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)

        def run_epoch(split):
            print(f"device: {self.device}")
            print(f"save interval: {config.ckpt_interval}")
            print(f"batch size: {config.batch_size}")
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers)

            losses = []
            writer = SummaryWriter(config.logdir)
            if progress_bar:
                pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            else:
                pbar = enumerate(loader)
            last_time = time.monotonic()
            printed = False
            running_loss = 0.0
            for it, dat in pbar:

                x = dat['sequence']
                y = dat['values'][0]

                # place data on the correct device

                x = encode(x)
                if x is None: #When x contains any N's, encode(x) will return none
                    continue
                
                x = x.to(self.device)
                y = y.to(self.device)
                

                # forward the model
                with torch.set_grad_enabled(is_train), torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
                    y_hat, loss = model(x, y, None)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                if is_train:
  
                    # backprop and update the parameters
                    scaler.scale(loss).backward()
                    # Update Optimizer
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()


                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            if not printed:
                                print("finished warmup!")
                                printed = True
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    if config.ckpt_interval is not None:
                        if it % (config.ckpt_interval//config.batch_size) == 0:
                            self.save_checkpoint(it)

                    current_time = time.monotonic()
                    delta_time = current_time - last_time
                    last_time = current_time
                                            

                    # report progress

                    running_loss += (loss.item() - running_loss)/min(it+1.0, 1000.0)
                    
                    if it%20 == 0:
                        print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}, running loss {running_loss:0.5f}, it/sec: {1.0/delta_time}")
                    
            if not is_train:
                logger.info("test loss: %f", np.mean(losses))

        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                run_epoch('test')
            self.save_checkpoint()
