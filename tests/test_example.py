import pathlib

import typst_pyimage


def test_lotka_volterra():
    here = pathlib.Path(__file__).resolve().parent
    filename = here / ".." / "examples" / "lotka_volterra.typ"
    typst_pyimage.compile(filename)
    typst_pyimage.watch(filename, timeout_s=2)
