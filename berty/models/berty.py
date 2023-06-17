assert __name__!='__main__', "'model.py' should only be imported"

import torch
import torch.nn as nn
import dnaset.bigwig_dataset as bigwig_dataset
import math
import logging
from torch.nn import functional as F

logger = logging.getLogger(__name__)

STATE_DICT_KEY_DEFAULT="state_dict"
CONFIG_KEY_DEFAULT="config"

class BERTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    use_z_residuals = False

    n_layer = 12
    n_head = 12
    n_embd = 768
    n_intermediate = None # will default to 4*n_embd
    initializer_range = 0.02

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    #just copying this directly from minGPT
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # dropouts force network to rely on a larger range of features
        # by disabling some of them randomly
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # linear layer for output
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
    
    def forward(self, x):
        # batch, idx, channel
        B, T, C= x.size()

        # generates our keys, values, and queries. Splits them so M[b,h,t,i] will obtain the ith entry
        # from the t'th element of the bth batch that's being fed into head h.
        # each matrix is size [B,num_heads, T, head_size]
        keys = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        queries = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        values = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        # take the dot product of each query with each key and normalize. 
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attention = (queries @ keys.transpose(-2,-1)) * (1.0 / math.sqrt(keys.size(-1)))
        
        # [b, nh, t1, t2] represents the similarity between query t1 and key t2. We want to make sure that
        # the sum of a given query's keys add up to 1, so we perform a softmax across axis 4
        attention = nn.functional.softmax(attention, dim=-1)

        # will randomly zero some features to force the model to learn from every feature
        attention = self.attn_drop(attention)
        
        # use attention to form linear combination of values based on attention scores for each entry
        # (B, nh, T, T,) * (B, nh, T, hs) -> (B, nh, T, hs)
        y = attention @ values

        #reassemble heads side-by-side
        y = y.transpose(1, 2).contiguous().view(B,T,C)

        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ a completely innocent Transformer block """

    def __init__(self,config):
        super().__init__()
        # LayerNorm will normalize elements across the last len(config.n_embd)
        # to have mean B and standard deviation G where B and G are learnable parameters
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)

        n_intermediate = config.n_intermediate
        if n_intermediate is None:
            n_intermediate = 4* config.n_embd

        self.use_z_residuals = config.use_z_residuals

        # From what I can tell, there's not too much intuition on
        # why this is done in this particular order. It just is
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, n_intermediate),
            nn.GELU(),
            nn.Linear(n_intermediate, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        #layer norm -> attention -> layer norm -> multi-layer-perceptron
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Berty(nn.Module):
    """
    A variation on BERT that perform regression
    """
    #loads model from state_dict
    def restore(self, checkpoint_path, key=STATE_DICT_KEY_DEFAULT):
        checkpoint = torch.load(checkpoint_path)
        if key is not None:
            checkpoint = checkpoint[key]
        self.load_state_dict(checkpoint)

    def restore_config_from_loaded_checkpoint(checkpoint, config_key=CONFIG_KEY_DEFAULT, state_dict_key=STATE_DICT_KEY_DEFAULT):
        config = checkpoint[config_key]
        state_dict = checkpoint[state_dict_key]
        model = BERTY(config)
        model.load_state_dict(state_dict)
        return model
    
    def recover_config(self):
        config = self.config
        config.vocab_size = self.tok_emb.vocab_size
        config.n_embd = self.tok_emb.embedding_dim
        config.block_size = self.block_size
        config.embd_pdrop = self.drop.p
        config.n_layer = len(self.blocks)
        # no obvious way to recover config.initializer_range, but this is not important
        # because we do need to initialize anything anymore.

        return config

    def get_save_dict(self, config_key=CONFIG_KEY_DEFAULT, state_dict_key=STATE_DICT_KEY_DEFAULT):
        return {
            config_key: self.config,
            state_dict_key: self.state_dict()
        }

    def save_to_checkpoint(self, checkpoint_path, config_key=CONFIG_KEY_DEFAULT, state_dict_key=STATE_DICT_KEY_DEFAULT):
        torch.save(self.get_save_dict(config_key, state_dict_key), checkpoint_path)
    
    def restore_config_from_checkpoint(checkpoint_path, config_key=CONFIG_KEY_DEFAULT, state_dict_key=STATE_DICT_KEY_DEFAULT):
        checkpoint = torch.load(checkpoint_path)
        return BERTY.restore_config_from_loaded_checkpoint(checkpoint, config_key, state_dict_key)

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(
            1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head. We're performing a regression task,
        # so self.head.out_dimension = 1
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 1, bias=True) 

        self.block_size = config.block_size
        self.apply(self._get_init_weights_fn(config))

        #We're going to use poisson loss because assay outputs can only be positive and many, many of them are near-zero
        self.softplus = nn.Softplus()
        self.opt_config_called = False
    
    # custom weight initializer. Probably there's some theoretical backing to this
    def _get_init_weights_fn(self, config):
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        return _init_weights

    def get_block_size(self):
        return self.block_size

    def get_features(self, idx):
        _, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted: {}, of {}".format(t, self.block_size)

        # forward the BERT model
        # each index maps to a (learnable) vector
        token_embddings = self.tok_emb(idx)
        # each position maps to a (learnable) vector
        position_embddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embddings + position_embddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        return x
    
    def forward(self, idx, targets=None, mask=None):
        assert idx.type() == torch.uint8, "Model only accepts torch index arrays"
        assert self.opt_config_called, "Please call self.configure_optimizers() and use result"
        x = self.get_features(idx)
        y_hat = self.softplus(self.head(x))

        # if we're given some desired targets, also calculate the loss
        loss = None
        y_hat = torch.squeeze(y_hat,2)
        assert all([d1==d2 for d1,d2 in zip(y_hat.shape, idx.shape)]), f"Shape mismatch between input {idx.shape} and output {y_hat.shape}"
        assert all(y_hat>0), "Some entries of y_hat < 0??"
        return y_hat
    
    def configure_optimizers(self):
        self.opt_config_called = True
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    # print("ok....",m)
                elif not pn.endswith('weight'):
                    no_decay.add(fpn)
                # :
                    # print('wtf is this: ', fpn)
                    # if pn.endswith('z_attn'):
                    #     print(m)
                    # print(isinstance(m, torch.nn.LayerNorm))
                    # print(isinstance(m, torch.nn.Linear))
                    # print(m)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # print("decay: ", str(decay))
        # print("no decay: ", str(no_decay))

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        
        # print("named params: ", str(param_dict.keys()))

        inter_params = decay & no_decay
        union_params = decay | no_decay
        # print(param_dict.keys())
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups)
        return optimizer



