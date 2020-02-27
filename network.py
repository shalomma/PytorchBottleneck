import torch


class FeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForward, self).__init__()
        torch.manual_seed(1234)
        self.hidden_sizes = hidden_sizes
        self.layers = torch.nn.ModuleList()
        hidden_sizes += [output_size]
        self.n_layers = len(hidden_sizes)
        prev = input_size
        for h in hidden_sizes:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hiddens = []
        for j, layer in enumerate(self.layers):
            if j != (self.n_layers - 1):
                x = torch.tanh(layer(x))
                hiddens.append(x)
            else:
                x = layer(x)
                hiddens.append(x)
        return x, hiddens
