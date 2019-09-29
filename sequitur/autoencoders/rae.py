# Third Party
import torch
import torch.nn as nn


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.num_features))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.output_dim = 2 * input_dim, output_dim

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # self.perceptrons = nn.ModuleList()
        # for _ in range(seq_len):
        #     self.perceptrons.append(nn.Linear(self.hidden_dim, output_dim))

        self.dense_layers = torch.rand(
            (self.hidden_dim, output_dim),
            dtype=torch.float,
            requires_grad=True
        )

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        # output_seq = torch.empty(
        #     self.seq_len,
        #     self.output_dim,
        #     dtype=torch.float
        # )
        # for index, perceptron in zip(range(self.seq_len), self.perceptrons):
        #     output_seq[index] = perceptron(x[index])
        #
        # return output_seq

        return torch.mm(x, self.dense_layers)


#########
# EXPORTS
#########


class RAE(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim=64):
        super(RAE, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, num_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
