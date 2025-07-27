import logging
import pathlib
import sys

from ._run import compile, watch


# argparse? click? fire? What are those?
# This is simpler for a trivial CLI like this one.
if (
    "-h" in sys.argv
    or "--help" in sys.argv
    or len(sys.argv) != 3
    or sys.argv[1] not in {"compile", "watch"}
):
    print("Usage is `python -m typst_pyimage [compile|watch] <filename>`")
    sys.exit(0)
_, command, filename = sys.argv
filename = pathlib.Path(filename).resolve()
try:
    logger = logging.getLogger("typst_pyimage")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if command == "compile":
        compile(filename, should_raise=False)
    elif command == "watch":
        watch(filename, should_raise=False)
    else:
        assert False
except KeyboardInterrupt:
    sys.exit(130)
