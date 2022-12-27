#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re, platform
from pkg_resources import packaging
from setuptools import setup, find_packages
from pathlib import Path

init = Path(__file__).parent / "lib" / "torchfsdd" / "__init__.py"
def load_meta(meta):
    with open(init, "r") as file:
        info = re.search(rf'^__{meta}__\s*=\s*[\'"]([^\'"]*)[\'"]', file.read(), re.MULTILINE).group(1)
        if not info:
            raise RuntimeError(f"Could not load {repr(meta)} metadata")
        return info

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

# Backports for importlib.metadata for Python versions < v3.8
install_requires = []
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    install_requires.append('importlib_metadata')

setup(
    name = load_meta("name"),
    version = load_meta("version"),
    author = load_meta("author"),
    author_email = load_meta("email"),
    description = 'A utility for wrapping the Free Spoken Digit Dataset into PyTorch-ready data set splits.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/eonu/torch-fsdd',
    project_urls = {
        'Documentation': 'https://torchfsdd.readthedocs.io/en/latest',
        'Bug Tracker': 'https://github.com/eonu/torch-fsdd/issues',
        'Source Code': 'https://github.com/eonu/torch-fsdd'
    },
    license = 'MIT',
    package_dir = {'': 'lib'},
    packages = find_packages(where='lib'),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English'
    ],
    python_requires = '>=3.6',
    install_requires = install_requires,
    extras_require = {
        'torch': [
            'torch>=1.8+cpu',
            'torchaudio>=0.8+cpu'
        ],
        'dev': [
            'torch>=1.8+cpu',
            'torchaudio>=0.8+cpu',
            'torchvision>=0.8',
            'sphinx',
            'numpydoc',
            'sphinx_rtd_theme',
            'sphinx-autobuild',
            'm2r2',
            'mistune==0.8.4',
            'Jinja2<3.1',
            'sphinx-version-warning',
            'pytest'
        ]
    }
)