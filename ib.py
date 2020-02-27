import numpy as np
import utils
import torch
from random import seed

from network import FeedForward
from train import Train, TrainConfig

np.random.seed(1234)
seed(1234)
torch.manual_seed(1234)


def tensor_casting(dataset):
    return torch.tensor(dataset.X, dtype=torch.float32), \
           torch.tensor(dataset.y, dtype=torch.float32), \
           torch.tensor(dataset.c, dtype=torch.long)


if '__main__' == __name__:

    trn, tst = utils.get_ib_data()

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
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    hidden_sizes = [10, 7, 5, 4, 3]
    net = FeedForward(input_size, hidden_sizes, output_size)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    cfg = TrainConfig(net, criterion, optimizer)
    train = Train(cfg)
    train.epochs = 5000
    train.run(data)
    train.plot_losses()
    train.plot_info_plan('train')
    train.plot_info_plan('test')
