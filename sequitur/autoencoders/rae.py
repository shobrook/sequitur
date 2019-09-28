# Third Party
import torch
import torch.nn as nn


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, num_features, embedding_size=64):
        super(Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=128,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_size))


class Bridge(nn.Module):
    def __init__(self, num_timesteps):
        super(Bridge, self).__init__()

        self.num_timesteps = num_timesteps

    def forward(self, x):
        x = x.repeat(self.num_timesteps, 1)

        return x.reshape((1, self.num_timesteps, 64))


class Decoder(nn.Module):
    def __init__(self, num_timesteps, output_dim=1, num_features=64):
        super(Decoder, self).__init__()

        self.num_timesteps, self.output_dim = num_timesteps, output_dim
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
        self.output_layers = nn.ModuleList(self.layers)

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
        for t, mlp in zip(range(self.num_timesteps), self.output_layers):
            output_seq[t] = mlp(x[t])

        return output_seq

        # return torch.mm(x, self.output)


#########
# EXPORTS
#########


class RAE(nn.Module):
    def __init__(self, num_timesteps, num_features, embedding_size=64):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(num_features)
        self.bridge = Bridge(num_timesteps)
        self.decoder = Decoder(num_timesteps, num_features, embedding_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bridge(x)
        x = self.decoder(x)

        return x
