import numpy as np
import torch
from random import seed
from torch.utils.data import DataLoader

from utils import get_ib_data
from dataset import IBDataset
from network import FeedForward
from train import Train, TrainConfig
from plotter import Plotter

np.random.seed(1234)
seed(1234)
torch.manual_seed(1234)


def tensor_casting(dataset):
    return torch.tensor(dataset.X, dtype=torch.float32), \
           torch.tensor(dataset.y, dtype=torch.float32), \
           torch.tensor(dataset.c, dtype=torch.long)


if '__main__' == __name__:

    data = dict()
    data['train'] = IBDataset(train=True)
    data['test'] = IBDataset(train=False)

    n_train = len(data['train'])
    n_test = len(data['test'])

    loader = dict()
    loader['train'] = DataLoader(data['train'], batch_size=n_train, shuffle=False)
    loader['test'] = DataLoader(data['test'], batch_size=n_test, shuffle=False)

    trn, tst = get_ib_data()

    x_train, y_train, c_train = tensor_casting(trn)
    x_test, y_test, c_test = tensor_casting(tst)

    data = dict()
    data['train'] = {}
    data['train']['samples'] = x_train
    data['train']['labels'] = y_train
    data['train']['class'] = c_train
    data['test'] = {}
    data['test']['samples'] = x_test
    data['test']['labels'] = y_test
    data['test']['class'] = c_test

    # setup
    input_size = 12
    output_size = 2
    hidden_sizes = [10, 7, 5, 4, 3]
    net = FeedForward(input_size, hidden_sizes, output_size)

    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    cfg = TrainConfig(net, criterion, optimizer)
    train = Train(cfg)
    train.epochs = 10000
    # train.n_classes = 2
    train.mi_cycle = 100
    train.run(data)

    plot = Plotter(train)
    plot.plot_losses()
    plot.plot_accuracy()
    plot.plot_info_plan('train')
    plot.plot_info_plan('test')
