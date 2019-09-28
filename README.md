# sequitur

`sequitur` is a Recurrent Autoencoder for sequence data that works out-of-the-box. It's easy to configure and only takes one line of code to use:

```python
from sequitur import QuickEncode

sequences = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
encoder, decoder, embeddings, f_loss  = QuickEncode(sequences, embedding_dim=2)

encoder([13,14,15,16]) # => [0.19, 0.84]
```

## Installation

> Requires Python 3.x and PyTorch 1.2.x

You can install `sequitur` with pip:

`$ pip install sequitur`

or download a compiled binary here.

## Usage

`sequitur` will learn how to represent arbitrarily long sequences as lower-dimensional, fixed-size vectors. `QuickEncode` implements an Encoder-Decoder architecture with an LSTM, making it optimal for sequences with long-term dependencies (e.g. time series data).

```python
from sequitur.autoencoders import RAE, SAE, VAE
```

## Architecture

<!--Provide proof that it's generally effective-->

---

https://github.com/szagoruyko/pytorchviz
https://github.com/RobRomijnders/AE_ts
https://github.com/erickrf/autoencoder
https://miro.medium.com/max/1400/1*sWc8g2yiQrOzntbVeGzbEQ.png
https://arxiv.org/pdf/1502.04681.pdf
