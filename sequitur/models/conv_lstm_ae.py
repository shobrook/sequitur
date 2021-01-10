# Third Party
import torch
import torch.nn as nn

# Local Modules
from sequitur.models.lstm_ae import LSTM_AE
from sequitur.models.conv_ae import CONV_AE


######
# MAIN
######


class CONV_LSTM_AE(nn.Module):
    def __init__(self, input_dims, encoding_dim, kernel, stride=1,
                 h_conv_channels=[1], h_lstm_channels=[]):
        super(CONV_LSTM_AE, self).__init__()

        self.input_dims = input_dims
        self.conv_enc_dim = sum(input_dims)

        self.conv_ae = CONV_AE(
            input_dims,
            self.conv_enc_dim,
            kernel,
            stride,
            h_channels=h_conv_channels
        )
        self.lstm_ae = LSTM_AE(
            self.conv_enc_dim,
            encoding_dim,
            h_lstm_channels
        )

    def encoder(self, x):
        n_elements, encodings = x.shape[0], []
        for i in range(n_elements):
            element = x[i].unsqueeze(0).unsqueeze(0)
            encodings.append(self.conv_ae.encoder(element))

        return self.lstm_ae.encoder(torch.stack(encodings))

    def decoder(self, x, seq_len):
        encodings = self.lstm_ae.decoder(x, seq_len)
        decodings = []
        for i in range(seq_len):
            decodings.append(torch.squeeze(self.conv_ae.decoder(encodings[i])))

        return torch.stack(decodings)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x
