import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from tqdm import tqdm

# parameters
root="./data/"
input_size = 28*28
hidden_size = 200
output_size = 10
num_hidden = 4
# end parameters

# hyperparameters
batch_size = 4
epochs = 100
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

        # hidden layers
        self.layers = [nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())]
        for i in range(num_hidden-1):
            stack = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            self.layers.append(stack)

        self.norm = nn.LayerNorm(hidden_size)

        # layer for labeling (uses last 3 hidden layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Sequential(nn.Linear(3*hidden_size, output_size), nn.Softmax(dim=1))

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.goodness_fn = goodness


    def forward(self, x):
        x = self.flatten(x)
        label_layers = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norm(x)
            if i != 0:
                label_layers.append(x)
        full_label_layer = torch.cat(label_layers, dim=1)
        probs = self.softmax(full_label_layer)
        return probs

    def predict(self, x):
        probs = self(x)
        return probs.argmax(1)

    def _train(self, x, y, sign):
        x = self.flatten(x)

        label_layers = []
        for i, layer in enumerate(self.layers):
            activations = layer(x)

            # if sign == 1, maximize goodness
            # if sign == -1, minimize goodness
            g = -1 * sign * self.goodness_fn(activations)

            self.optimizer.zero_grad()
            g.backward()
            self.optimizer.step()

            x = self.norm(activations)
            x = x.detach()
            if i != 0:
                label_layers.append(x)

        # for i in range(len(label_layers)):
        #     label_layers[i] = label_layers[i].detach()

        full_label_layer = torch.cat(label_layers, dim=1)
        probs = self.softmax(full_label_layer)
        loss = self.loss_fn(probs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, dataloader):
        size = len(dataloader.dataset) #number of samples
        train_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            loss = self._train(X, y, sign=1)
            # loss = self.loss_fn(pred, y)

            train_loss += loss.item()
            if batch % 10 == 0:
                loss = loss.item()
                current = (batch+1) * len(X) # (len(X) is the batch size)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        return train_loss

    def test(self, dataloader, name="Test"):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0


        with torch.no_grad():
            for X, y in dataloader:
                probs = self(X)
                test_loss += self.loss_fn(probs, y).item()
                correct += (probs.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"{name} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, correct







if __name__ == "__main__":
    torch.manual_seed(0)

    net = FF()
    train_loss = net.train(train_loader)
    net.test(valid_loader, "Validation")
    net.test(test_loader)
