# Third Party
import torch.nn as nn


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        num_layers = len(layer_dims) - 1
        layers = []
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index], layer_dims[index + 1])

            if h_activ and index < num_layers - 1:
                layers.extend([layer, h_activ])
            elif out_activ and index == num_layers - 1:
                layers.extend([layer, out_activ])
            else:
                layers.append(layer)

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        num_layers = len(layer_dims) - 1
        layers = []
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index], layer_dims[index + 1])

            if h_activ and index < num_layers - 1:
                layers.extend([layer, h_activ])
            else:
                layers.append(layer)

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


######
# MAIN
######


class LINEAR_AE(nn.Module):
    def __init__(self, input_dim, encoding_dim, h_dims=[], h_activ=nn.Sigmoid(),
                 out_activ=nn.Tanh()):
        super(LINEAR_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ,
                               out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, list(reversed(h_dims)),
                               h_activ)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
