import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
)
from torch import nn
from torch.nn import Module
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=input_dim, out_features=64, bias=True)
        self.hidden2 = nn.Linear(in_features=64, out_features=1, bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        return x

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(100000,5120)
    
    def forward(self, x):
        x = self.embedding(x)
        return x
