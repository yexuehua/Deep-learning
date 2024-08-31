import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq):
        super().__init__()
        self.max_seq = max_seq
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        odd_i = torch.arange(1, self.d_model, 2).float()
        even_denominator = torch.pow(10000, even_i/self.d_model)
        odd_denominator = torch.pow(10000, (odd_i-1)/self.d_model)
        position = torch.arange(self.max_seq).reshape(self.max_seq, 1)
        even_PE = torch.sin(position/even_denominator)
        odd_PE = torch.cos(position/odd_denominator)
        PE = torch.concat((even_PE, odd_PE), dim=1)
        return PE
    
pe = PositionalEncoding(6,10)
pe.forward()