# sequitur

`sequitur` is a Recurrent Autoencoder (RAE) for sequence data that works out-of-the-box. It's easy to configure and only takes one line of code to use.

```python
from sequitur import QuickEncode

sequences = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
encoder, decoder, embeddings, f_loss  = QuickEncode(sequences, embedding_dim=2)

encoder([13,14,15,16]) # => [0.19, 0.84]
```

<!--An autoencoder will learn how to-->`sequitur` will learn how to represent sequences of any length as lower-dimensional, fixed-size vectors. This can be useful for finding patterns among sequences, clustering, converting sequences into inputs for a machine learning algorithm, and dimensionality reduction.

## Installation

> Requires Python 3.X and PyTorch 1.2.X

You can download a compiled binary [here](https://github.com/shobrook/sequitur/) or install `sequitur` with pip:

`$ pip install sequitur`

## API

#### `sequitur.QuickEncode(sequences, embedding_dim, logging=False, lr=1e-3, epochs=100)`

Lets you train an autoencoder with just one line of code. This wraps a PyTorch implementation of an Encoder-Decoder architecture with an LSTM, making this optimal for sequences with long-term dependencies (e.g. time series data).

**Parameters**

- `sequences`: A list (or tensor) of shape `[num_seqs, seq_len, num_features]` representing your training set of sequences.
  - Each sequence should have the same length, `seq_len`, and contain a sequence of vectors of size `num_features`.
  - If `num_features=1`, then you can input a list of shape `[num_seqs, seq_len]` instead.
- `embedding_dim`: Size of the vector encodings you want to create.
- `logging`: Boolean for whether you want logging statements to be printed during training.
- `lr`: Learning rate for the autoencoder.
- `epochs`: Number of epochs to train for.

**Returns**

- `encoder`: The trained encoder as a PyTorch module.
  - Takes as input a tensor of shape `[seq_len, num_features]` representing a sequence where each element is a vector of size `num_features`.
- `decoder`: The trained decoder as a PyTorch module.
  - Takes as input a tensor of shape `[embedding_dim]` representing an encoded sequence.
- `embeddings`: A tensor of shape `[num_seqs, embedding_dim]` which holds the learned vector encodings of each sequence in the training set.
- `f_loss`: The final mean squared error of the autoencoder on the training set.

#### `sequitur.autoencoders.RAE(hyperparams)`

To-Do.

#### `sequitur.autoencoders.SAE(hyperparams)`

To-Do.

### `sequitur.autoencoders.VAE(hyperparams)`

To-Do.

## Contributing

`QuickEncode` is useful for rapid prototyping but doesn't give you much control over the model and training process. For that, you can import the RAE implementation itself from `sequitur.autoencoders`.

`sequitur` not only implements an RAE but also a Stacked Autoencoder (SAE) and a WIP Variational Autoencoder (VAE). If you've implemented a sequence autoencoder, or know of an implementation, please feel free to add it to the codebase and open a pull request. With enough autoencoders, I can turn `sequitur` into a small PyTorch extension library.

<!--Provide proof that it's generally effective-->

<!-- https://github.com/szagoruyko/pytorchviz
https://github.com/RobRomijnders/AE_ts
https://github.com/erickrf/autoencoder
https://miro.medium.com/max/1400/1*sWc8g2yiQrOzntbVeGzbEQ.png
https://arxiv.org/pdf/1502.04681.pdf -->
