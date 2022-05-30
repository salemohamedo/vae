from torch import nn
from torchvision import datasets

class Encoder(nn.Module):
    def __init__(self, x_dims, z_dims, hidden_dims=[128, 64, 36, 18]):
        super().__init__()
        layers = nn.ModuleList()
        hidden_dims.insert(0, x_dims)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU()
            ))
        self.encoder = nn.Sequential(*layers)
        self.fc_means = nn.Linear(hidden_dims[-1], z_dims)
        self.fc_vars = nn.Linear(hidden_dims[-1], z_dims)

    def forward(self, x):
        outs = self.encoder(x)
        means = self.fc_means(outs)
        vars = self.fc_vars(outs)
        return means, vars

enc = Encoder(512, 10)
print("Hi")

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(z):
        return
        
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(x):
        return