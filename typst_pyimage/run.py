import pathlib
import re
import shutil
import subprocess
import textwrap
import time
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt


pyimage_re = re.compile(r"pyimage\(\s*\"([^\"]*)\"")
pyimageinit_re = re.compile(r"pyimageinit\(\s*\"([^\"]*)\"")


def _malformed_fig(msg: str):
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        f"Malformed: {msg}",
        horizontalalignment="center",
        verticalalignment="center",
    )
    return fig


def _make_fig(code: str):
    plt.figure()  # Don't overwrite our current figure.
    try:
        with matplotlib.rc_context():
            exec(code, {})
    except Exception as e:
        fig = _malformed_fig(str(e))
    else:
        # Not the plt.figure() from above, just in case the code creates a new figure.
        fig = plt.gcf()
    return fig


def _make_images(
    filepath: pathlib.Path, dirpath: pathlib.Path, figcache: Optional[dict]
) -> None:
    for file in dirpath.iterdir():
        if file.name.endswith(".png"):
            file.unlink()
    with open(filepath) as f:
        contents = f.read()
    inits = list(pyimageinit_re.finditer(contents))
    if len(inits) == 0:
        init_code = ""
        multiple_inits = False
    elif len(inits) == 1:
        [init_code] = inits[0].groups()
        init_code = textwrap.dedent(init_code).strip()
        multiple_inits = False
    else:
        multiple_inits = True
    for i, match in enumerate(pyimage_re.finditer(contents)):
        if multiple_inits:
            fig = _malformed_fig("Cannot have multiple #pyimageinit directives")
        else:
            i = i + 1
            [code] = match.groups()
            code = textwrap.dedent(code).strip()
            code = init_code + "\n" + code  # pyright: ignore
            if figcache is None:
                fig = _make_fig(code)
            else:
                try:
                    fig = figcache[code]
                except KeyError:
                    fig = _make_fig(code)
        fig.savefig(dirpath / f"{i}.png")


def _get_file_time(filepath: pathlib.Path) -> int:
    return filepath.stat().st_mtime_ns


def _initial(filename: Union[str, pathlib.Path], figcache: Optional[dict]):
    installpath = pathlib.Path(__file__).resolve().parent
    filepath = pathlib.Path(filename).resolve()
    dirpath = filepath.parent / ".typst_pyimage"
    dirpath.mkdir(exist_ok=True)
    shutil.copy(installpath / "pyimage.typ", dirpath / "pyimage.typ")
    _make_images(filepath, dirpath, figcache)
    return filepath, dirpath


def watch(filename: Union[str, pathlib.Path], timeout_s: Optional[int] = None):
    figcache = {}
    filepath, dirpath = _initial(filename, figcache)
    del filename
    start_time = time.time()
    file_time = last_time = _get_file_time(filepath)
    keep_running = True
    need_update = False
    process = subprocess.Popen(["typst", "watch", str(filepath)])
    try:
        while keep_running:
            if need_update:
                last_time = file_time
                _make_images(filepath, dirpath, figcache)
            time.sleep(0.1)
            file_time = _get_file_time(filepath)
            need_update = file_time > last_time
            if timeout_s is not None:
                keep_running = time.time() < start_time + timeout_s
    except KeyboardInterrupt:
        pass
    finally:
        process.kill()


def compile(filename: Union[str, pathlib.Path]):
    filepath, _ = _initial(filename, figcache=None)
    subprocess.run(["typst", "compile", str(filepath)])
