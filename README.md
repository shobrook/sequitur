# sequitur

`sequitur` is a library that lets you create and train an autoencoder for sequential data in just two lines of code. It implements three different autoencoder architectures in PyTorch, and a predefined training loop. `sequitur` is ideal for working with sequential data ranging from single and multivariate time series to videos, and is geared for those who want to get started _quickly_ with autoencoders.

```python
import torch
from sequitur.models import LINEAR_AE
from sequitur import quick_train

train_seqs = [torch.randn(4) for _ in range(100)] # 100 sequences of length 4
encoder, decoder, _, _ = quick_train(LINEAR_AE, train_seqs, encoding_dim=2, denoise=True)

encoder(torch.randn(4)) # => torch.tensor([0.19, 0.84])
```

Each autoencoder learns to represent input sequences as lower-dimensional, fixed-size vectors. This can be useful for finding patterns among sequences, clustering sequences, or converting sequences into inputs for other algorithms.

<img src="./img/demo.png" />

## Installation

> Requires Python 3.X and PyTorch 1.2.X

You can install `sequitur` with `pip`:

```bash
$ pip install sequitur
```

## Getting Started

### 1. Prepare your data

First, you need to prepare a set of example sequences to train an autoencoder on. This training set should be a list of `torch.Tensor`s, where each tensor has shape `[num_elements, *num_features]`. So, if each example in your training set is a sequence of 10 5x5 matrices, then each example would be a tensor with shape `[10, 5, 5]`.

### 2. Choose an autoencoder

Next, you need to choose an autoencoder model. If you're working with sequences of numbers (e.g. time series) or 1D vectors (e.g. word vectors), then you should use the `LINEAR_AE` or `LSTM_AE` model. For sequences of 2D matrices (e.g. videos) or 3D matrices (e.g. fMRI scans), you'll want to use `CONV_LSTM_AE`. Each model is a PyTorch module, and can be imported like so:

```python
from sequitur.models import CONV_LSTM_AE
```

More details about each model are in the "Models" section below.

### 3. Train the autoencoder

From here, you can either initialize the model yourself and write your own training loop, or import the `quick_train` function and plug in the model, training set, and desired encoding size, like so:

```python
import torch
from sequitur.models import CONV_LSTM_AE
from sequitur import quick_train

train_set = [torch.randn(10, 5, 5) for _ in range(100)]
encoder, decoder, _, _ = quick_train(CONV_LSTM_AE, train_set, encoding_dim=4)
```

After training, `quick_train` returns the `encoder` and `decoder` models, which are PyTorch modules that can encode and decode new sequences. These can be used like so:

```python
x = torch.randn(10, 5, 5)
z = encoder(x) # Tensor with shape [4]
x_prime = decoder(z) # Tensor with shape [10, 5, 5]
```

## API

#### Training your Model

**`quick_train(model, train_set, encoding_dim, verbose=False, lr=1e-3, epochs=50, denoise=False, **kwargs)`**

Lets you train an autoencoder with just one line of code. Useful if you don't want to create your own training loop. Training involves learning a vector encoding of each input sequence, reconstructing the original sequence from the encoding, and calculating the loss (mean-squared error) between the reconstructed input and the original input. The autoencoder weights are updated using the Adam optimizer.

<!--If `denoise=True`, then each input sequence is injected with Gaussian noise before being fed into the autoencoder. The autoencoder is then trained to reconstruct the original undistorted input.-->

**Parameters:**

- `model` _(torch.nn.Module)_: Autoencoder model to train (imported from `sequitur.models`)
- `train_set` _(list)_: List of sequences (each a `torch.Tensor`) to train the model on; has shape `[num_examples, seq_len, *num_features]`
- `encoding_dim` _(int)_: Desired size of the vector encoding
- `verbose` _(bool, optional (default=False))_: Whether or not to print the loss at each epoch
- `lr` _(float, optional (default=1e-3))_: Learning rate
- `epochs` _(int, optional (default=50))_: Number of epochs to train for
<!--- `denoise` _(bool, optional=(default=False))_: If `True`, converts autoencoder into a [Denoising Autoencoder (DAE)](https://en.wikipedia.org/wiki/Autoencoder#Regularized_Autoencoders)-->
- `**kwargs`: Parameters to pass into `model` when it's instantiated

**Returns:**

- `encoder` _(torch.nn.Module)_: Trained encoder model; takes a sequence (as a tensor) as input and returns an encoding of the sequence as a tensor of shape `[encoding_dim]`
- `decoder` _(torch.nn.Module)_: Trained decoder model; takes an encoding (as a tensor) and returns a decoded sequence
- `encodings` _(list)_: List of tensors corresponding to the final vector encodings of each sequence in the training set
- `losses` _(list)_: List of average MSE values at each epoch

### Models

Every autoencoder inherits from `torch.nn.Module` and has an `encoder` attribute and a `decoder` attribute, both of which also inherit from `torch.nn.Module`.

#### Sequences of Numbers

**`LINEAR_AE(input_dim, encoding_dim, h_dims=[], h_activ=torch.nn.Sigmoid(), out_activ=torch.nn.Tanh())`**

Consists of fully-connected layers stacked on top of each other. Can only be used if you're dealing with sequences of numbers, not vectors or matrices.

<img src="./img/linear_ae.png" />

**Parameters:**

- `input_dim` _(int)_: Size of each input sequence
- `encoding_dim` _(int)_: Size of the vector encoding
- `h_dims` _(list, optional (default=[]))_: List of hidden layer sizes for the encoder
- `h_activ` _(torch.nn.Module or None, optional (default=torch.nn.Sigmoid()))_: Activation function to use for hidden layers; if `None`, no activation function is used
- `out_activ` _(torch.nn.Module or None, optional (default=torch.nn.Tanh()))_: Activation function to use for the output layer in the encoder; if `None`, no activation function is used

**Example:**

To create the autoencoder shown in the diagram above, use the following arguments:

```python
from sequitur.models import LINEAR_AE

model = LINEAR_AE(
  input_dim=10,
  encoding_dim=4,
  h_dims=[8, 6],
  h_activ=None,
  out_activ=None
)

x = torch.randn(10) # Sequence of 10 numbers
z = model.encoder(x) # z.shape = [4]
x_prime = model.decoder(z) # x_prime.shape = [10]
```

#### Sequences of 1D Vectors

**`LSTM_AE(input_dim, encoding_dim, h_dims=[], h_activ=torch.nn.Sigmoid(), out_activ=torch.nn.Tanh())`**

Autoencoder for sequences of vectors which consists of stacked LSTMs. Can be trained on sequences of varying length.

<img src="./img/lstm_ae.png" />

**Parameters:**

- `input_dim` _(int)_: Size of each sequence element (vector)
- `encoding_dim` _(int)_: Size of the vector encoding
- `h_dims` _(list, optional (default=[]))_: List of hidden layer sizes for the encoder
- `h_activ` _(torch.nn.Module or None, optional (default=torch.nn.Sigmoid()))_: Activation function to use for hidden layers; if `None`, no activation function is used
- `out_activ` _(torch.nn.Module or None, optional (default=torch.nn.Tanh()))_: Activation function to use for the output layer in the encoder; if `None`, no activation function is used

**Example:**

To create the autoencoder shown in the diagram above, use the following arguments:

```python
from sequitur.models import LSTM_AE

model = LSTM_AE(
  input_dim=3,
  encoding_dim=7,
  h_dims=[64],
  h_activ=None,
  out_activ=None
)

x = torch.randn(10, 3) # Sequence of 10 3D vectors
z = model.encoder(x) # z.shape = [7]
x_prime = model.decoder(z, seq_len=10) # x_prime.shape = [10, 3]
```

#### Sequences of 2D/3D Matrices

**`CONV_LSTM_AE(input_dims, encoding_dim, kernel, stride=1, h_conv_channels=[1], h_lstm_channels=[])`**

Autoencoder for sequences of 2D or 3D matrices/images, loosely based on the CNN-LSTM architecture described in _[Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/pdf/1503.08909.pdf)._ Uses a CNN to create vector encodings of each image in an input sequence, and then an LSTM to create encodings of the sequence of vectors.

<img src="./img/conv_lstm_ae.png" />

**Parameters:**

- `input_dims` _(tuple)_: Shape of each 2D or 3D image in the input sequences
- `encoding_dim` _(int)_: Size of the vector encoding
- `kernel` _(int or tuple)_: Size of the convolving kernel; use tuple to specify a different size for each dimension
- `stride` _(int or tuple, optional (default=1))_: Stride of the convolution; use tuple to specify a different stride for each dimension
- `h_conv_channels` _(list, optional (default=[1]))_: List of hidden channel sizes for the convolutional layers
- `h_lstm_channels` _(list, optional (default=[]))_: List of hidden channel sizes for the LSTM layers

**Example:**

```python
from sequitur.models import CONV_LSTM_AE

model = CONV_LSTM_AE(
  input_dims=(50, 100),
  encoding_dim=16,
  kernel=(5, 8),
  stride=(3, 5),
  h_conv_channels=[4, 8],
  h_lstm_channels=[32, 64]
)

x = torch.randn(22, 50, 100) # Sequence of 22 50x100 images
z = model.encoder(x) # z.shape = [16]
x_prime = model.decoder(z, seq_len=22) # x_prime.shape = [22, 50, 100]
```
