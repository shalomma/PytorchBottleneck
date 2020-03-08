import torch
from torch.utils import data
from torchvision import datasets
import scipy.io as sio


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, download=False, randomize=False):
        super(MNIST, self).__init__(root, train, download)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.targets = self.targets.to(device)
        self.data = self.data / 255.0
        self.data = self.data.view(-1, 28 * 28).to(device)

        if randomize:
            idx_rnd = torch.randperm(len(self.targets))
            self.targets = self.targets[idx_rnd]

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        return img, targets


class IBDataset(data.Dataset):
    def __init__(self, train=True, ratio=0.8):
        super(IBDataset, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_, targets_ = self.load_data(train, ratio)

        self.targets = targets_.to(device)
        self.target = targets_.to(device)
        self.data = data_.to(device)

    @staticmethod
    def load_data(train, ratio):
        d = sio.loadmat('./dataset/var_u.mat')
        x = d['F']
        y = d['y'][0]

        r = ratio if train else (1 - ratio)
        n = int(r * len(x))
        if train:
            x = x[:n, :]
            y = y[:n]
        else:
            x = x[-n:, :]
            y = y[-n:]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, target = self.data[index], self.target[index]
        return sample, target
