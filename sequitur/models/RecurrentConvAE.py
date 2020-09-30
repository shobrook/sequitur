# Third Party
import torch.nn as nn

# Local Modules
from .RecurrentAE import RecurrentAE
from .ConvAE import ConvAE


######
# MAIN
######


class RecurrentConvAE(nn.Module):
    def __init__(self, input_dims, encoding_dim, in_channels,
                 h_conv_channels=[], h_lstm_channels=[], kernel=None,
                 stride=None):
        super(RecurrentConvAE, self).__init__()

        self.input_dims = input_dims
        self.conv_enc_dim = sum(input_dims) * in_channels

        self.conv_ae = ConvAE(
            input_dims,
            self.conv_enc_dim,
            in_channels,
            h_conv_channels,
            kernel,
            stride
        )
        self.recurrent_ae = RecurrentAE(
            self.conv_enc_dim,
            encoding_dim,
            h_lstm_channels
        )

    def encoder(self, x):
        n_elements, encodings = x.shape[0], []
        for i in range(n_elements):
            element = x[i].reshape((
                1,
                1,
                *x[i].shape
            ))
            encodings.append(self.conv_ae.encoder(element))

        return self.recurrent_ae.encoder(torch.stack(encodings))

    def decoder(self, x, seq_len):
        encodings = self.recurrent_ae.decoder(torch.squeeze(x), seq_len)
        decodings = []
        for i in range(seq_len):
            decodings.append(self.conv_ae.decoder(encodings[i]))

        return torch.stack(decodings)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encode(x)
        x = self.decode(x, seq_len)

        return x
