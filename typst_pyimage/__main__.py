import sys

from .run import compile as compile, watch as watch


if len(sys.argv) != 3:
    raise RuntimeError("Usage is `python -m typst_pyimage <command> <file>`")
_, command, filename = sys.argv
if command == "watch":
    watch(filename)
elif command == "compile":
    compile(filename)
else:
    raise RuntimeError(f"Invalid command {command}")
