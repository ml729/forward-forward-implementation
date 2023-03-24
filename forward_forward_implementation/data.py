import torch

class NegativeMNIST(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return len(self.data)
