from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("musicue")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+dev"
