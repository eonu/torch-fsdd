.. _torchfsdd:

Generating data set splits
==========================

To use TorchFSDD to create :class:`torch:torch.utils.data.Dataset` data set objects for FSDD,
you first need to create a data set generator using the :class:`torchfsdd.TorchFSDDGenerator` class.

This class first downloads the data set from the GitHub repository (if not already downloaded),
then allows you to generate data splits (full, train/test, or train/validation/test) by automatically
selecting which files belong to which partition.

Each data set split is represented by a :class:`torchfsdd.TorchFSDD` object, which is a wrapper for :class:`torch:torch.utils.data.Dataset`.
For each split, the data set generator initializes one of these data sets and passes on the files that compose that split,
along with any transformations that should be applied to the recordings.

.. autoclass:: torchfsdd.TorchFSDDGenerator
    :members:

.. autoclass:: torchfsdd.TorchFSDD
    :members:

Transformations
===============

While many transformations can be applied to audio data, this package only includes a transformation
for trimming silence from the start or end of each audio recording. The implementation of this transformation
is exactly the same as the `trimming utility on the FSDD repository <https://github.com/Jakobovski/free-spoken-digit-dataset/blob/master/utils/trimmer.py>`_,
but for PyTorch tensors, and assuming a normalized signal (since :py:func:`torchaudio:torchaudio.load` automatically normalizes).

Trimming silence
----------------

.. autoclass:: torchfsdd.TrimSilence
    :members: