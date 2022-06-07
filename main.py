from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from argparse import ArgumentParser

from typing import List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist'], default='mnist')
parser.add_argument('--num-epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--latent-size', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=64)

args = parser.parse_args()
print(vars(args))

def prepare_dataloaders(dataset: str = 'mnist') -> Tuple[DataLoader, DataLoader]:
    if dataset == 'mnist':
        train_dataset = datasets.MNIST('./data',
                                       transform=transforms.ToTensor(),
                                       download=True,
                                       train=True)

        val_dataset = datasets.MNIST('./data',
                                      transform=transforms.ToTensor(),
                                      download=True,
                                      train=False)
    else:
        raise ValueError(f"Dataset: {dataset} not supported!")
    
    train_dataset.data = train_dataset.data[:2]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, val_loader

class Encoder(nn.Module):
    def __init__(self, x_dims: int, z_dims: int, hidden_dims: List) -> None:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(start_dim=1)
        outs = self.encoder(x)
        means = self.fc_means(outs)
        log_vars = self.fc_vars(outs)
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, x_dims: int, z_dims: int, hidden_dims: List) -> None:
        super().__init__()
        layers = nn.ModuleList()
        hidden_dims.insert(0, z_dims)
        hidden_dims.append(x_dims)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU()
            ))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        out = self.decoder(z)
        return out.unflatten(-1, (1, 28, 28))
        
class VAE(nn.Module):
    def __init__(self, x_dims: int, z_dims: int, hidden_dims: list = [128, 64, 36, 18]) -> None:
        super().__init__()
        self.encoder = Encoder(x_dims, z_dims, hidden_dims)
        self.decoder = Decoder(x_dims, z_dims, hidden_dims[::-1])
        self.z_dims = z_dims
        self.normal = torch.distributions.Normal(0, 1)

    def reparameterize(self, means: torch.Tensor, log_vars: torch.Tensor) -> torch.Tensor:
        eps = self.normal.sample(log_vars.shape).to(device)
        vars = torch.exp(log_vars)
        std = vars.sqrt()
        self.kl = (-0.5)*(log_vars - vars - means**2 + 1).sum()
        return means + std*eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means, log_vars = self.encoder(x)
        z = self.reparameterize(means, log_vars)
        return self.decoder(z)

class VAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_x: torch.Tensor, true_x: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        return torch.square(pred_x - true_x).sum() + kl

def train(model: VAE, train_loader: DataLoader, criterion: VAELoss, optim: torch.optim.Optimizer) -> None:
    model.train()
    for x, _ in train_loader:
        x = x.to(device)
        optim.zero_grad()
        x_pred = model(x)
        loss = criterion(x_pred, x, model.kl)
        print(loss)
        loss.backward()
        optim.step()

@torch.no_grad()
def eval(model: VAE, val_loader: DataLoader, criterion: VAELoss) -> torch.Tensor:
    model.eval()
    total_loss = 0
    for x, _ in val_loader:
        x = x.to(device)
        x_pred = model(x)
        loss = criterion(x_pred, x, model.kl)
        total_loss += loss
    return total_loss / len(val_loader)

train_loader, val_loader = prepare_dataloaders(args.dataset)

vae = VAE(x_dims=784, z_dims=args.latent_size).to(device)
optim = torch.optim.Adam(vae.parameters(), lr=args.lr)
criterion = VAELoss()

for i in range(args.num_epochs):
    train(vae, train_loader, criterion, optim)
    # val_loss = eval(vae, val_loader, criterion)
    # print(f"Eval loss: {val_loss:.4f}")

