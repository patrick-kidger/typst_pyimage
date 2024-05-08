import argparse
import os

from .run import compile as compile, watch as watch


def parse_args(args):
    filepath = args.filepath

    # Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found")

    extra_args = args.extra_args.replace("~", os.path.expanduser("~")).split()

    return filepath, extra_args


def _compile_wrapper(args):
    try:
        filepath, extra_args = parse_args(args)
    except Exception as e:
        print(e)
        return

    compile(filepath, extra_args)


def _watch_wrapper(args):
    try:
        filepath, extra_args = parse_args(args)
    except Exception as e:
        print(e)
        return

    watch(filepath, extra_args)


program_description = (
    "Typst extension, adding support for generating"
    + " figures using inline Python code."
)


# Create the parser
parser = argparse.ArgumentParser(
    prog="typst_pyimage",
    description=program_description,
)

# Create subparsers
subparser = parser.add_subparsers(help="sub-command help")

# Create subparsers for the subcommands
subparser_compile = subparser.add_parser(
    "compile", aliases=["c"], help="Compile the typst file"
)
subparser_watch = subparser.add_parser(
    "watch", aliases=["w"], help="Watch the typst file"
)

# Set the subparsers to call the appropriate function
subparser_compile.set_defaults(func=_compile_wrapper)
subparser_watch.set_defaults(func=_watch_wrapper)

# Required positional argument
parser.add_argument(
    "filepath",
    action="store",
    type=str,
    default=".",
    help="The path to the typst file to be compiled/watched",
)

# Optional argument
parser.add_argument(
    "-a",
    "--extra-args",
    type=str,
    help="Extra arguments to be passed to typst",
)

# Parse the arguments
args = parser.parse_args()
args.func(args)
