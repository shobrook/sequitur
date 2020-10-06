# Third Party
import torch.nn as nn


#######
# UNITS
#######


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, in_channels, input_dim, dim):
        super(UnFlatten, self).__init__()

        self.dim = dim
        self.in_channels, self.input_dim = in_channels, input_dim

    def forward(self, input):
        # return input.reshape((1, self.in_channels, 2, 2, 2))

        # output = torch.zeros(1, self.in_channels, 2, 2, 2)
        # output[:, :, :1, :1, :1] = input.reshape((1, self.in_channels, 1, 1, 1))
        #
        # return output

        input_dims = tuple(self.input_dim for _ in range(self.dim))
        return input.reshape((1, self.in_channels, *input_dims))


class PadExternal(nn.Module):
    def __init__(self, padded_input_dim):
        super(PadExternal, self).__init__()

        self.targ_input_dim = padded_input_dim

    def forward(self, input):
        # Input can either be a rectangular prism or a cube
        return input # TODO


class PadInternal(nn.Module):
    def __init__(self, padding, dim):
        super(PadInternal, self).__init__()

        self.padding, self.dim = padding, dim

    def _calculate_padded_dim(self, dim_size):
        return ((dim_size - 1) * self.padding) + dim_size

    def forward(self, input):
        stride = self.padding + 1

        if self.dim == 3:
            _, in_channels, m, n, o = input.shape
            input = input.reshape((in_channels, m, n, o))
            output = torch.zeros(
                in_channels,
                self._calculate_padded_dim(m),
                self._calculate_padded_dim(n),
                self._calculate_padded_dim(o)
            )
            output[:, ::stride, ::stride, ::stride] = input
        elif self.dim == 2:
            _, in_channels, m, n = input.shape
            input = input.reshape((in_channels, m, n))
            output = torch.zeros(
                in_channels,
                self._calculate_padded_dim(m),
                self._calculate_padded_dim(n)
            )
            output[:, ::stride, ::stride] = input

        return output.reshape((1, *output.shape))


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

    def forward(self, input):
        output = self.conv(input)
        output = self.relu(output)

        return output


class DeConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dim):
        super(DeConvUnit, self).__init__()

        # TODO: Handle dim == 1
        deconv = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        self.deconv = deconv(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.deconv(input)
        output = self.relu(output)

        return output


###########
# UTILITIES
###########


def compute_output_dim(num_layers, input_dim, kernel, stride):
    if num_layers == 1:
        return input_dim

    # Guide to convolutional arithmetic: https://arxiv.org/pdf/1603.07285.pdf
    out_dim = floor((input_dim - kernel) / stride) + 1
    return compute_output_dim(num_layers - 1, out_dim, kernel, stride)


def compute_flattened_output_dim(out_dim, out_channels, dim):
    return (out_dim ** dim) * out_channels


######
# MAIN
######


class CONV_AE(nn.Module):
    def __init__(self, input_dims, encoding_dim, in_channels=1, h_channels=[],
                 kernel=None, stride=None):
        super(CONV_AE, self).__init__()

        # TODO: Handle kernel and/or stride == None
            # Automatically calculate maximum kernel_size and minimum stride
        # TODO: Handle hidden_channels == []
        # TODO: Handle kernel and stride as 2D or 3D tuples

        conv_dim = len(input_dims)
        all_channels = [in_channels] + h_channels
        num_layers = len(all_channels)

        # Compute input and output shapes
        padded_input_dim = max(input_dims)
        out_dim = compute_output_dim(num_layers, padded_input_dim, kernel,
                                     stride)
        flattened_out_dim = compute_flattened_output_dim(out_dim,
                                                         all_channels[-1],
                                                         conv_dim)

        # Construct encoder and decoder units
        encoder_layers = [PadExternal(padded_input_dim=0)] # QUESTION: Why is this here?
        decoder_layers = [
            nn.Linear(encoding_dim, flattened_out_dim),
            UnFlatten(all_channels[-1], out_dim, conv_dim)
        ]
        for index in range(num_layers - 1):
            conv_layer = ConvUnit(
                in_channels=all_channels[index],
                out_channels=all_channels[index + 1],
                kernel=kernel,
                stride=stride,
                dim=conv_dim
            )
            upsample_layer = PadInternal(padding=stride - 1, dim=conv_dim)
            deconv_layer = DeConvUnit(
                in_channels=all_channels[-index - 1],
                out_channels=all_channels[-index - 2],
                kernel=kernel,
                dim=conv_dim
            )

            encoder_layers.append(conv_layer)
            decoder_layers.extend([upsample_layer, deconv_layer])

        encoder_layers.extend([
            Flatten(),
            # nn.ReLU(),
            nn.Linear(flattened_out_dim, encoding_dim)
        ])
        # decoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # BUG: Encoder CNN adds padding to non-square/cubic images. Thus, the
        # decoder CNN will recreate a padded image. You cannot just reshape
        # this to match the original image dimensions.

    def forward(self, input):
        input = self.encoder(input)
        return self.decoder(input)
