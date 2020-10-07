# Standard Library
from math import floor
from functools import reduce

# Third Party
import torch.nn as nn


#######
# UNITS
#######


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1).squeeze()


class UnFlatten(nn.Module):
    def __init__(self, in_channels, input_dims):
        super(UnFlatten, self).__init__()

        self.in_channels, self.input_dims = in_channels, input_dims

    def forward(self, x):
        return x.reshape((1, self.in_channels, *self.input_dims))


# class PadExternal(nn.Module):
#     def __init__(self, padded_input_dim):
#         super(PadExternal, self).__init__()
#
#         self.targ_input_dim = padded_input_dim
#
#     def forward(self, input):
#         # Input can either be a rectangular prism or a cube
#         return input # TODO
#
#
# class PadInternal(nn.Module):
#     def __init__(self, padding, dim):
#         super(PadInternal, self).__init__()
#
#         self.padding, self.dim = padding, dim
#
#     def _calculate_padded_dim(self, dim_size):
#         return ((dim_size - 1) * self.padding) + dim_size
#
#     def forward(self, input):
#         stride = self.padding + 1
#
#         if self.dim == 3:
#             _, in_channels, m, n, o = input.shape
#             input = input.reshape((in_channels, m, n, o))
#             output = torch.zeros(
#                 in_channels,
#                 self._calculate_padded_dim(m),
#                 self._calculate_padded_dim(n),
#                 self._calculate_padded_dim(o)
#             )
#             output[:, ::stride, ::stride, ::stride] = input
#         elif self.dim == 2:
#             _, in_channels, m, n = input.shape
#             input = input.reshape((in_channels, m, n))
#             output = torch.zeros(
#                 in_channels,
#                 self._calculate_padded_dim(m),
#                 self._calculate_padded_dim(n)
#             )
#             output[:, ::stride, ::stride] = input
#
#         return output.reshape((1, *output.shape))


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dim):
        super(ConvUnit, self).__init__()

        # TODO: Handle dim == 1
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class DeConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dim):
        super(DeConvUnit, self).__init__()

        # TODO: Handle dim == 1
        deconv = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        self.deconv = deconv(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride
        )
        self.relu = nn.ReLU()

    def forward(self, x, output_size):
        x = self.deconv(x, output_size=output_size)
        # x = self.relu(x)

        return x


###########
# UTILITIES
###########


def compute_output_dim(num_layers, input_dim, kernel, stride, out_dims=[]):
    if not num_layers:
        return out_dims

    # Guide to convolutional arithmetic: https://arxiv.org/pdf/1603.07285.pdf
    out_dim = floor((input_dim - kernel) / stride) + 1
    out_dims.append(out_dim)

    return compute_output_dim(num_layers - 1, out_dim, kernel, stride, out_dims)


######
# MAIN
######


class CONV_AE(nn.Module):
    def __init__(self, input_dims, encoding_dim, kernel, stride, in_channels=1,
                 h_channels=[1]):
        super(CONV_AE, self).__init__()

        conv_dim = len(input_dims)
        all_channels = [in_channels] + h_channels
        num_layers = len(all_channels) - 1

        if isinstance(kernel, int):
            kernel = (kernel, ) * conv_dim
        if isinstance(stride, int):
            stride = (stride, ) * conv_dim

        out_dims = []
        for i, k, s in zip(input_dims, kernel, stride):
            out_dims.append(compute_output_dim(num_layers, i, k, s, []))
        out_dims = [input_dims] + list(zip(*out_dims))

        self.out_dims = out_dims[::-1]
        out_dims = self.out_dims[0]
        flat_dim = all_channels[-1] * reduce(lambda x, y: x * y, out_dims)

        # Construct encoder and decoder units
        encoder_layers = []
        self.decoder_layers = nn.ModuleList([
            nn.Linear(encoding_dim, flat_dim),
            UnFlatten(all_channels[-1], out_dims)
        ])
        for index in range(num_layers):
            conv_layer = ConvUnit(
                in_channels=all_channels[index],
                out_channels=all_channels[index + 1],
                kernel=kernel,
                stride=stride,
                dim=conv_dim
            )
            deconv_layer = DeConvUnit(
                in_channels=all_channels[-index - 1],
                out_channels=all_channels[-index - 2],
                kernel=kernel,
                stride=stride,
                dim=conv_dim
            )

            encoder_layers.append(conv_layer)
            self.decoder_layers.append(deconv_layer)

        encoder_layers.extend([Flatten(),
                               nn.Linear(flat_dim, encoding_dim)])
        self.encoder = nn.Sequential(*encoder_layers)

    def decoder(self, x):
        for index, layer in enumerate(self.decoder_layers):
            if isinstance(layer, DeConvUnit):
                x = layer(x, output_size=self.out_dims[1:][index - 2])
            else:
                x = layer(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
