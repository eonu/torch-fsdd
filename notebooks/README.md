# TorchFSDD Example

This directory contains a Jupyter Notebook demonstration showing how TorchFSDD can be used to quickly generate PyTorch datasets and data loaders,
and use these to train a deep recurrent neural network.

**The rendered notebook can be viewed [here](https://nbviewer.jupyter.org/github/eonu/torch-fsdd/tree/master/notebooks/demo.ipynb).**

## Running the notebook

To run the notebook yourself, you will have to first install a number of dependencies.<br/>These are found in [`requirements.txt`](./requirements.txt).

You should run the following command in the root directory of the repository to install them.

```console
pip install -r notebooks/requirements.txt
```

Once these are installed, you can run the [`demo.ipynb`](./ipynb) notebook using Jupyter Notebook.

```console
jupyter notebook notebooks/demo.ipynb
```

## Model

The particular network that we use is a PyTorch implementation of the DeepGRU[[1]](#references) architecture, found in [`model.py`](./model.py).

## References

<table>
  <tbody>
    <tr>
      <td>[1]</td>
      <td>
        <a href="https://arxiv.org/ftp/arxiv/papers/1810/1810.12514.pdf">Mehran Maghoumi & Joseph J. LaViola Jr. <b>"DeepGRU: Deep Gesture Recognition Utility"</b> <em>Advances in Visual Computing, 14th International Symposium on Visual Computing, ISVC 2019</em>, Proceedings, Part I, 16-31.</a>
      </td>
    </tr>
  </tbody>
</table>