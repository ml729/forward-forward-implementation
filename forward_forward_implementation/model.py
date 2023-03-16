import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch import torchvision
from torch.torchvision import transforms
from tqdm import tqdm

# parameters
root="./data/"
input_size = 10
hidden_size = 5
output_size = 1
num_hidden = 2
# end parameters

# hyperparameters
batch_size = 4
epochs = 5
lr = 1e-3
# end hyperparameters




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

og_train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)

train_set, valid_set = random_split(og_train_set, [50000,10000])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)



test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)



def goodness(a):
    return (a**2).sum()

class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = [nn.Sequential(nn.Linear(28*28, 512), nn.ReLU())]
        for i in range(num_hidden):
            stack = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
            self.layers.append(stack)
        self.layers.append(nn.Linear(512, 10))

        self.norm = nn.LayerNorm(512)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        self.goodness_fn = goodness


    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = self.norm(layer(x))

        return x

    def _train(self, x, sign):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            activations = layer(x)

            # if sign == 1, maximize goodness
            # if sign == -1, minimize goodness
            g = -1 * sign * self.goodness_fn(activations)

            self.optimizer.zero_grad()
            g.backward()
            self.optimizer.step()

            x = self.norm(x)

        logits = self.layers[-1](x)
        return logits

    def train(self, dataloader):
        size = len(dataloader.dataset) #number of samples
        train_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            pred = self(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if batch % 10 == 0:
                loss = loss.item()
                current = (batch+1) * len(X) # (len(X) is the batch size)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        return train_loss

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0


        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, correct







if __name__ == "__main__":
    torch.manual_seed(0)

    net = TestNN()
    train_loss = net.train(train_loader)

