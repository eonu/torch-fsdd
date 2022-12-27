# Changelog

## [1.0.0](https://github.com/eonu/torch-fsdd/releases/tag/v1.0.0)

#### Major changes

- Rework `setup.py` to use package extras and fetch metadata from `__init__.py`. ([#15](https://github.com/eonu/torch-fsdd/pull/15))
- Add `dev` package extra for development (including running docs and tests). ([#15](https://github.com/eonu/torch-fsdd/pull/15))

#### Minor changes

- Switch to CircleCI. ([#15](https://github.com/eonu/torch-fsdd/pull/15))
- Add CI for Python versions (3.6 - 3.10). ([#15](https://github.com/eonu/torch-fsdd/pull/15))
- Fix `Jinja2<3.1` version for RTD (see [readthedocs/readthedocs.org/issues/9038](https://github.com/readthedocs/readthedocs.org/issues/9038)). ([#15](https://github.com/eonu/torch-fsdd/pull/15))
- Fix broken ReadTheDocs link — thanks @black-puppydog. ([#13](https://github.com/eonu/torch-fsdd/pull/13))

## [0.1.2](https://github.com/eonu/torch-fsdd/releases/tag/v0.1.2)

#### Major changes

- Don't allow zero `test_size` (see #7). ([#11](https://github.com/eonu/torch-fsdd/pull/11))
- Change `extra_requires` to `extras_require` in `setup.py`. ([#9](https://github.com/eonu/torch-fsdd/pull/9))<br/>(this is why `pip install torchfsdd[torch]` didn't work!)

#### Minor changes

- Bump package development status to beta. ([#10](https://github.com/eonu/torch-fsdd/pull/10))
- Swap `torch` and `torchaudio` links in `README.md`. ([#8](https://github.com/eonu/torch-fsdd/pull/8))

## [0.1.1](https://github.com/eonu/torch-fsdd/releases/tag/v0.1.1)

#### Major changes

- Add `torch` and `torchaudio` dependencies to `docs/requirements.txt`. ([#4](https://github.com/eonu/torch-fsdd/pull/4))
- Upgrade minimum package dependency versions: ([#3](https://github.com/eonu/torch-fsdd/pull/3))
  - `torch` (>= 1.8)
  - `torchaudio` (>= 0.8)
  - `torchvision` (>= 0.9) - only used in tests and notebooks

#### Minor changes

- Remove platform-dependent `torch`, `torchaudio` and `torchvision` installation for tests. ([#1](https://github.com/eonu/torch-fsdd/pull/1))
- Move Travis badges to separate `README.md` section. ([#2](https://github.com/eonu/torch-fsdd/pull/2))

## [0.1.0](https://github.com/eonu/torch-fsdd/releases/tag/v0.1.0)

#### Major changes

Nothing, initial release!