import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, num_features):
        super(Encoder, self).__init__()

        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, 64))


class Bridge(nn.Module):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def forward(self, x):
        x = x.repeat(self.num_timesteps, 1)

        return x.reshape((1, self.num_timesteps, 64))


class Decoder(nn.Module):
    def __init__(self, num_timesteps, output_dim=1, num_features=64):
        self.num_timesteps = num_timesteps

        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=num_features,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.layers = [nn.Linear(128, output_dim) for _ in range(num_timesteps)]
        self.output_layers = nn.ModuleList(self.output)

        # self.output = torch.tensor(128, output_dim, requires_grad=True)

    def forward(self, x):
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.num_timesteps, 128))

        output_seq = torch.empty(
            self.num_timesteps,
            self.output_dim,
            dtype=torch.float
        )
        for t, mlp in zip(self.num_timesteps, self.output_layers):
            output_seq[t] = mlp(x[t])

        return output_seq

        # return torch.mm(x, self.output)
