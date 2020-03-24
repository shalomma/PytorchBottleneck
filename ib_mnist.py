import numpy as np
import torch
from torch.utils.data import DataLoader
from random import seed

from dataset import MNIST
from network import FeedForward
from train_mnist import Train, TrainConfig
from plotter import Plotter


np.random.seed(1234)
seed(1234)
torch.manual_seed(1234)


if '__main__' == __name__:

    data = dict()
    data['train'] = MNIST('./dataset', train=True, download=True, randomize=False)
    data['test'] = MNIST('./dataset', train=False)

    loader = dict()
    loader['train'] = torch.utils.data.DataLoader(data['train'], batch_size=60000, shuffle=False)
    loader['test'] = torch.utils.data.DataLoader(data['test'], batch_size=10000, shuffle=False)

    # setup
    input_size = 28 * 28
    output_size = 10
    hidden_sizes = [784, 1024, 1024, 20, 20, 20, 10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'to device: {device}')
    net = FeedForward(input_size, hidden_sizes, output_size).to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    cfg = TrainConfig(net, criterion, optimizer)
    train = Train(cfg)
    train.epochs = 4000
    train.mi_cycle = 20
    train.run(loader)
    train.dump()

    plot = Plotter(train)
    plot.plot_losses()
    plot.plot_accuracy()
    plot.plot_info_plan('train')
    plot.plot_info_plan('test')
