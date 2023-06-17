import torch

class Linear(torch.nn.Module):
    def __init__(self):
        '''
        Accepts index array (torch.uint8) corresponding to the base pairs of a 
        dna sequence segment. Will pass one-hot encoding through a single linear layer
        '''
        self.linear = torch.nn.Linear(4,1)
        def encoder(x):
            return torch.nn.one_hot(x,4)
        self.encode = encoder
        self.softplus = torch.nn.Softplus()
    
    def forward(self,x):
        y_hat = self.linear(self.encode(x))
        return self.softplus(y_hat)
