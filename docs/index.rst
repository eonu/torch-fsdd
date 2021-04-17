.. TorchFSDD documentation master file, created by
   sphinx-quickstart on Sat Dec 28 19:22:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About
=====

The `Free Spoken Digit Dataset <https://github.com/Jakobovski/free-spoken-digit-dataset>`_ is an open data set consisting of audio recordings of various
individuals speaking the digits from 0-9, with 50 recordings of each digit per individual.

The data set can be though of as an audio version of the popular `MNIST data set <https://en.wikipedia.org/wiki/MNIST_database>`_
which consists of hand-written digits. However, the fact that the data consists of recordings
of different length makes it more challenging to deal with than the fixed-size images of MNIST.

Models based on recurrent neural networks that can be implemented in PyTorch are a common approach
for this task, and TorchFSDD aims to provide an interface to FSDD for such neural networks in PyTorch,
by providing a :class:`torch:torch.utils.data.Dataset` wrapper that is ready to be used with a :class:`torch:torch.utils.data.DataLoader`.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: About TorchFSDD

   self
   changelog.rst

.. toctree::
   :maxdepth: 1
   :caption: Using TorchFSDD

   sections/torchfsdd.rst

Documentation Search and Index
==============================

* :ref:`search`
* :ref:`genindex`