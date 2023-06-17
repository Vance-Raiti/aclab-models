import torch
import math



class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = torch.nn.Conv1d(
            in_channels,out_channels,kernel_size=3,stride=1,padding=1, bias=False
        )
        self.conv2 = torch.nn.Conv1d(
            out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False
        )

    def forward(self,x):
        if (self.in_channels == self.out_channels):
          y = self.conv1(x)
          y = torch.nn.functional.relu(y)
          y = self.conv2(y)
          y = x + y
          return torch.nn.functional.relu(y)
        else:
          x = self.conv1(x)
          y = torch.nn.functional.relu(x)
          y = self.conv2(y)
          y = x + y
          return torch.nn.functional.relu(y)

class Model(torch.nn.Module):
    def __init__(self,n_blocks,n_steps,in_channels):
        '''
            Crude model that attempts to learn a bigWig file from input Fasta file

            args:
                n_blocks - number of convolution "blocks" that the model will use. The architecture of these blocks was copied directly
                from EC414 Hw 10
                n_steps - number of times we double the number of channels. Doubling will be distributed linearly
                throughout the layers
                in_channels - number of channels the initial input has
        '''


        super().__init__()

        self.layer_out_channels = [in_channels*2**(math.floor((n_steps)*i/n_blocks)) for i in range(n_blocks)]
        self.blocks = [self.make_layer(self.layer_out_channels[0],self.layer_out_channels[0])]
        
        for i in range(1,n_blocks):
            self.blocks.append(self.make_layer(self.layer_out_channels[i-1],self.layer_out_channels[i]))
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.linear = torch.nn.Linear(round(in_channels*2**(n_steps-1)),1)
        self.softplus = torch.nn.Softplus()
        def encode(x):
            return torch.nn.functional.one_hot(x,4)
        self.encoder = encode
    
    def forward(self,x):
        '''
            Forward pass for our experiment model
            
            args:
                seq - (8,n) tensor of a one-hot encoded genome sequence
            
            output:
                (1,n) tensor of predictions
        '''
        assert x.type()==torch.uint8, "Can only pass index array (torch.uint8) to this model"
        seq = self.encode(x)
        seq = self.blocks(seq)
        y_hat = self.linear(torch.transpose(seq,0,1))
        return self.softplus(torch.squeeze(torch.transpose(y_hat,0,1),0))
       
        

    def make_layer(self, in_channels, out_channels):
        layers = [
            ConvolutionBlock(in_channels,out_channels),
            ConvolutionBlock(out_channels,out_channels)
        ]
        return torch.nn.Sequential(*layers)





