import argparse
import os

from .run import compile as compile
from .run import watch as watch

# Create the parser
parser = argparse.ArgumentParser(
    prog="typst_pyimage",
    description="Typst extension, adding support for generating figures using inline Python code",
)

# Required positional argument
parser.add_argument(
    "filepath",
    action="store",
    type=str,
    help="The path to the typst file to be compiled/watched",
)

# Boolean flags
# The user can only choose one of these flags at a time
group = parser.add_mutually_exclusive_group()
group.add_argument("-c", "--compile", action="store_false", help="Compile the typst file")
group.add_argument("-w", "--watch", action="store_false", help="Watch the typst file")

# Optional argument
parser.add_argument(
    "-a",
    "--extra-args",
    type=str,
    help="Extra arguments to be passed to typst",
)

# Parse the arguments
args = parser.parse_args()

filepath = args.filepath
do_compile = args.compile
do_watch = args.watch
extra_args = args.extra_args.replace("~", os.path.expanduser("~")).split()

if do_compile:
    watch(filepath, extra_args)
elif do_watch:
    compile(filepath, extra_args)
