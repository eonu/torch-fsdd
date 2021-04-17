#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import platform
from pkg_resources import packaging
from setuptools import setup, find_packages

VERSION = '0.1.1'

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

# Backports for importlib.metadata for Python versions < v3.8
install_requires = []
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    install_requires.append('importlib_metadata')

setup(
    name = 'torchfsdd',
    version = VERSION,
    author = 'Edwin Onuonga',
    author_email = 'ed@eonu.net',
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
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
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
    extra_requires = {'torch': ['torch>=1.8+cpu', 'torchaudio>=0.8+cpu']}
)