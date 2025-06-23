import pathlib
import sys

from ._run import run


_, filename = sys.argv
run(pathlib.Path(filename).resolve())
