import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import pickle

from data import NegativeMNIST

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
lr = 1e-1
# end hyperparameters


torch.manual_seed(3)

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

# Create negative data

def create_mask():
    ## Create mask by iteratively blurring a random tensor and thresholding it
    ## Kernel for blurring
    kernel = torch.tensor([1/4, 1/2, 1/4], dtype=torch.float32)
    h_kernel = kernel.view(1, 1, 1, 3)
    h_blur_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), stride=1, padding=(0,1), bias=False)
    h_blur_layer.weight.data = h_kernel
    v_kernel = kernel.view(1, 1, 3, 1)
    v_blur_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), stride=1, padding=(1, 0), bias=False)
    v_blur_layer.weight.data = v_kernel

    def nblur(n):
        def blur(x):
            for _ in range(n):
                x = h_blur_layer(x)
                x = v_blur_layer(x)
            return x
        return blur

    ## Mask
    rand_tensor = torch.rand(train_set[0][0].shape)
    blurred_tensor = nblur(7)(rand_tensor)
    mask_tensor = (blurred_tensor > 0.5).float()
    # plt.imshow(mask_tensor.squeeze(), cmap='gray')
    # plt.show()
    ## Save
    torch.save(mask_tensor, "data/mask.pt")
    return mask_tensor

# if os.path.isfile("mask.pt"):
#     mask_tensor = torch.load("mask.pt")
# else:
#     mask_tensor = create_mask()

# Sample pairs of training set samples


def create_dataset():
    if os.path.isfile("mask.pt"):
        mask = torch.load("mask.pt")
    else:
        mask = create_mask()
    rev_mask = 1 - mask

    sampler1 = torch.utils.data.RandomSampler(train_set, replacement=False, num_samples=50000)
    sampler2 = torch.utils.data.RandomSampler(train_set, replacement=False, num_samples=50000)

    data = []
    for a, b in zip(sampler1, sampler2):
        if a == b: continue
        da = train_set[a][0]
        ya = train_set[a][1]
        db = train_set[b][0]
        yb = train_set[b][1]
        dnew = mask * da + rev_mask * db
        data.append(dnew)
    negative_train_set = NegativeMNIST(data)
    with open('data/negative_mnist.pickle', 'wb') as f:
        pickle.dump(negative_train_set, f)
    return negative_train_set


def get_negative_loader():
    if os.path.isfile("data/negative_mnist.pickle"):
        with open('data/negative_mnist.pickle', 'rb') as f:
            negative_train_set = pickle.load(f)
    else:
        negative_train_set = create_dataset()

    negative_train_loader = torch.utils.data.DataLoader(negative_train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return negative_train_loader




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

    def _train(self, pos_x, neg_x):
        avg_good = 0
        avg_bad = 0
        for x, sign in zip((pos_x, neg_x), (1, -1)):
            x = self.flatten(x)

            label_layers = []
            for i, layer in enumerate(self.layers):
                activations = layer(x)

                # if sign == 1, maximize goodness
                # if sign == -1, minimize goodness
                goodness = self.goodness_fn(activations)
                g = -1 * sign * goodness
                if sign == 1:
                    avg_good += goodness
                else:
                    avg_bad += goodness

                self.optimizer.zero_grad()
                g.backward()
                self.optimizer.step()

                x = self.norm(activations)
                x = x.detach()
                if i != 0:
                    label_layers.append(x)
        avg_good /= len(self.layers)
        avg_bad /= len(self.layers)
        return avg_good, avg_bad

        # for i in range(len(label_layers)):
        #     label_layers[i] = label_layers[i].detach()

        # full_label_layer = torch.cat(label_layers, dim=1)
        # probs = self.softmax(full_label_layer)
        # loss = self.loss_fn(probs, y)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return loss
        return

    def _train_softmax(self, x, y):
        x = self.flatten(x)
        label_layers = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norm(x)
            if i != 0:
                label_layers.append(x)
        full_label_layer = torch.cat(label_layers, dim=1)
        probs = self.softmax(full_label_layer)
        loss = self.loss_fn(probs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def train(self, dataloader, negative_dataloader):
        print("Running Forward-Forward")
        size = len(dataloader.dataset) #number of samples
        train_loss = 0
        train_good = 0
        train_bad = 0
        # for batch, (X, y) in enumerate(dataloader):
        #     self._train(X)
        #     # loss = self.loss_fn(pred, y)

        #     # train_loss += loss.item()
        #     if batch % 10 == 0:
        #         # loss = loss.item()
        #         current = (batch+1) * len(X) # (len(X) is the batch size)
        #         print(f"[{current:>5d}/{size:>5d}]")


        for batch, ((X, y), neg_X) in enumerate(zip(dataloader, negative_dataloader)):
            goodness, badness = self._train(X, neg_X)
            # loss = self.loss_fn(pred, y)
            train_good += goodness
            train_bad += badness

            # train_loss += loss.item()
            if batch % 10 == 0:
                # loss = loss.item()
                goodness = goodness
                badness = badness
                current = (batch+1) * len(X) # (len(X) is the batch size)
                print(f"good: {goodness:>7f}, bad: {badness:>7f} [Training samples: {current:>5d}/{size:>5d}]")


        print("Training softmax layer")
        for batch, (X, y) in enumerate(dataloader):
            loss = self._train_softmax(X, y)

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

    negative_train_loader = get_negative_loader()
    net = FF()
    train_loss = net.train(train_loader, negative_train_loader)
    net.test(valid_loader, "Validation")
    net.test(test_loader)
