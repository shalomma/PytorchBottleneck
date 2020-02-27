import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from random import seed

from network import FeedForward
from train_mnist import Train, TrainConfig

np.random.seed(1234)
seed(1234)
torch.manual_seed(1234)


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, download=False):
        super(MNIST, self).__init__(root, train, download)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.data / 255.0
        self.data = self.data.view(-1, 28 * 28).to(device)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target


class IBDataset(DataLoader):

    def __init__(self, dataset):
        super(IBDataset, self).__init__()
        self.x, self.y, self.c = self.tensor_casting(dataset)

    @staticmethod
    def tensor_casting(dataset):
        return torch.tensor(dataset.X, dtype=torch.float), \
               torch.tensor(dataset.Y), \
               torch.tensor(dataset.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'samples': self.x[idx],
            'labels': self.y[idx],
            'class': self.c[idx]
        }

        return sample


if '__main__' == __name__:

    loader = dict()

    data = dict()
    data['train'] = MNIST('./dataset', train=True, download=True)
    data['test'] = MNIST('./dataset', train=False)

    loader['train'] = torch.utils.data.DataLoader(data['train'], batch_size=2048, shuffle=True)
    loader['test'] = torch.utils.data.DataLoader(data['test'], batch_size=2048, shuffle=True)

    # setup
    input_size = 28 * 28
    output_size = 10
    hidden_sizes = [1024, 20, 20, 20, 20]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'to device: {device}')
    net = FeedForward(input_size, hidden_sizes, output_size).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    cfg = TrainConfig(net, criterion, optimizer)
    train = Train(cfg)
    train.epochs = 1000
    train.run(loader)
    train.plot_losses()
    # train.plot_info_plan('train')
    # train.plot_info_plan('test')
