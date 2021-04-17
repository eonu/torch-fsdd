__version__ = '0.1.1'

import importlib, platform
from pkg_resources import packaging
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    import importlib_metadata as metadata
else:
    from importlib import metadata

MIN_TORCH_VERSION = '1.8'
MIN_TORCHAUDIO_VERSION = '0.8'

def check_package(pkg, min_version, url):
    """Checks whether a specified package has been installed,
    and whether the installed version meets a specified minimum.

    Parameters
    ----------
    pkg: str
        Name of the package.

    min_version: str
        Minimum version for the package, e.g. `1.8`.

    url: str
        Package installation page URL (for help).
    """
    try:
        importlib.import_module(pkg)
    except ImportError:
        msg = ("Could not find a valid installation of '{pkg}' (>={min_version}), which TorchFSDD depends on.\n"
        "Visit {url} for more instructions on installing this package.").format(pkg=pkg, url=url, min_version=min_version)
        raise ModuleNotFoundError(msg)

    installed_version = metadata.version(pkg)
    if packaging.version.parse(installed_version) < packaging.version.parse(min_version):
        msg = ("Could not find a compatible installation of '{pkg}' (>={min_version}), which TorchFSDD depends on - got version {installed_version}.\n"
        "Visit {url} for more instructions on installing this package.").format(pkg=pkg, url=url, min_version=min_version, installed_version=installed_version)
        raise ImportWarning(msg)

# Check that the minimum dependency versions are installed
check_package('torch', MIN_TORCH_VERSION, url='https://pytorch.org/')
check_package('torchaudio', MIN_TORCHAUDIO_VERSION, url='https://github.com/pytorch/audio')

# Import classes from the package
from .dataset import TorchFSDD, TorchFSDDGenerator
from .helpers import TrimSilence

__all__ = ['TorchFSDD', 'TorchFSDDGenerator', 'TrimSilence']