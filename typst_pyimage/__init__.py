import importlib.metadata

from ._run import compile as compile, watch as watch


__version__ = importlib.metadata.version("typst_pyimage")
