<h1 align="center">
    TorchFSDD
</h1>

<p align="center">
    <em>A utility for wrapping the Free Spoken Digit Dataset into PyTorch-ready data set splits.</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/torchfsdd">
        <img src="https://img.shields.io/pypi/v/torchfsdd?style=flat-square" alt="PyPI"/>
    </a>
    <a href="https://pypi.org/project/torchfsdd">
        <img src="https://img.shields.io/pypi/pyversions/torchfsdd?style=flat-square" alt="PyPI - Python Version"/>
    </a>
    <a href="https://raw.githubusercontent.com/eonu/torchfsdd/master/LICENSE">
        <img src="https://img.shields.io/pypi/l/torchfsdd?style=flat-square" alt="PyPI - License"/>
    </a>
    <a href="https://torch-fsdd.readthedocs.io/en/latest/">
        <img src="https://readthedocs.org/projects/torch-fsdd/badge/?version=latest&style=flat-square" alt="Read The Docs - Documentation">
    </a>
</p>

## About

The [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) is an open data set consisting of audio recordings of various individuals speaking the digits from 0-9, with 50 recordings of each digit per individual.

The data set can be though of as an audio version of the popular [MNIST data set](https://en.wikipedia.org/wiki/MNIST_database) which consists of hand-written digits. However, the fact that the data consists of recordings of different length makes it more challenging to deal with than the fixed-size images of MNIST.

Models based on recurrent neural networks that can be implemented in PyTorch are a common approach
for this task, and TorchFSDD aims to provide an interface to FSDD for such neural networks in PyTorch, by providing a [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) wrapper that is ready to be used with a [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

## Build status

| `master` | `dev` |
| -------- | ------|
| [![Travis Build (Master)](https://img.shields.io/travis/com/eonu/torch-fsdd?logo=travis&style=flat-square)](https://travis-ci.com/github/eonu/torch-fsdd) | [![Travis Build (Development)](https://img.shields.io/travis/com/eonu/torch-fsdd/dev?logo=travis&style=flat-square)](https://travis-ci.com/github/eonu/torch-fsdd) |

## Examples

```python
from torchfsdd import TorchFSDDGenerator, TrimSilence
from torchaudio.transforms import MFCC
from torchvision.transforms import Compose

# Create a transformation pipeline to apply to the recordings
transforms = Compose([
    TrimSilence(threshold=1e-6),
    MFCC(sample_rate=8e3, n_mfcc=13)
])

# Fetch the latest version of FSDD and initialize a generator with those files
fsdd = TorchFSDDGenerator(version='master', transforms=transforms)

# Create a Torch dataset for the entire dataset from the generator
full_set = fsdd.full()
# Create two Torch datasets for a train-test split from the generator
train_set, test_set = fsdd.train_test_split(test_size=0.1)
# Create three Torch datasets for a train-validation-test split from the generator
train_set, val_set, test_set = fsdd.train_val_test_split(test_size=0.15, val_size=0.15)
```

A more complete example can be found [here](./notebooks), showing how TorchFSDD can be used to train a neural network.

## Installation and Usage

You can install TorchFSDD using `pip`.

```console
pip install torchfsdd
```

**Note**: TorchFSDD assumes you have the following packages already installed (along with Python v3.6+).

- [`torch`](https://github.com/pytorch/audio) (>= 1.8.0)
- [`torchaudio`](https://github.com/pytorch/pytorch) (>= 0.8.0)

Since there are many different possible configurations when installing PyTorch (e.g. CPU or GPU, CUDA version), we leave this up to the user instead of specifying particular binaries to install alongside TorchFSDD.

Make sure you have `torch` and `torchaudio` versions that are [compatible](https://github.com/pytorch/audio#dependencies)!

> If you _really_ wish to install `torch` and `torchaudio` together with TorchFSDD automatically, the following will install CPU-only versions of both dependencies.
>
> ```console
> pip install torchfsdd[torch]
> ```

## Documentation

Documentation for the package is available on [Read The Docs](https://torchfsdd.readthedocs.io/en/latest).

## Contributors

All contributions to this repository are greatly appreciated. Contribution guidelines can be found [here](/CONTRIBUTING.md).

<table>
	<thead>
		<tr>
			<th align="center">
                <a href="https://github.com/eonu">
                    <img src="https://avatars0.githubusercontent.com/u/24795571?s=460&v=4" alt="Edwin Onuonga" width="60px">
                    <br/><sub><b>Edwin Onuonga</b></sub>
                </a>
                <br/>
                <a href="mailto:ed@eonu.net">✉️</a>
                <a href="https://eonu.net">🌍</a>
			</th>
			<!-- Add more <th></th> blocks for more contributors -->
		</tr>
	</thead>
</table>

---

<p align="center">
  <b>TorchFSDD</b> &copy; 2021-2022, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> License.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>