# Contributing

As TorchFSDD is an open source library, any contributions from the community are greatly appreciated. This document details the guidelines for making contributions to TorchFSDD.

## Reporting issues

Prior to reporting an issue, please ensure:

- [ ] You have used the search utility provided on GitHub issues to look for similar issues.
- [ ] You have checked the documentation (for the version of TorchFSDD you are using).
- [ ] You are using the latest version of TorchFSDD (if possible).

## Making changes to TorchFSDD

- **Add tests**: Your pull request won't be accepted if it doesn't have any tests.

- **Document any change in behaviour**: Make sure the README and all other relevant documentation is kept up-to-date.

- **Create topic branches**: Will not pull from your master branch!

- **One pull request per feature**: If you wish to add more than one new feature, please make multiple pull requests.

- **Meaningful commit messages**: Make sure each individual commit in your pull request has a meaningful message.

- **De-clutter commit history**: If you had to make multiple intermediate commits while developing, please squash them before making your pull request.
  Or add a note on the PR specifying to squash and merge your changes when ready to be merged.

### Making pull requests

Please make new branches based on the current `dev` branch, and merge your PR back into `dev` (making sure you have fetched the latest changes).

### Installing dependencies

If you intend to help contribute to TorchFSDD, you will need some additional dependencies for running tests, notebooks and generating documentation.

You can specify the `dev` extra when installing TorchFSDD to do this.

```console
pip install torch-fsdd[dev]
```

If installing a TorchFSDD from a local directory, you can use `pip install -e .` from within that directory, or `pip install -e .[xxx]` to install with extras.

Note that on some shells you may have to use quote marks, e.g. `pip install -e ".[xxx]"`.

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](/LICENSE) that covers this repository.
