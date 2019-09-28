# Third Party
import torch.nn as nn


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, 20)
        self.linear2 = nn.Linear(20, hidden_dim)

        self.hidden_activ = nn.Sigmoid()
        self.output_activ = nn.Tanh()

    def forward(self, x):
        x = self.hidden_activ(self.linear1(x))
        x = self.output_activ(self.linear2(x))

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, 20)
        self.linear2 = nn.Linear(20, input_dim)

        self.hidden_activ = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_activ(self.linear1(x))
        x = self.linear2(x)

        return x


#########
# EXPORTS
#########


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
