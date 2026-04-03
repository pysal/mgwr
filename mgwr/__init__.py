import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import gwr
from . import sel_bw
from . import diagnostics
from . import kernels

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("mgwr")
