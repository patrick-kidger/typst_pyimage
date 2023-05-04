import pathlib
import re
import shutil
import subprocess
import textwrap
import time
from typing import Optional, Tuple, Union

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


def _make_fig(code: str, scope: dict):
    plt.figure()  # Don't overwrite our current figure.
    scope = dict(scope)  # make a copy
    try:
        with matplotlib.rc_context():
            exec(code, scope)
    except Exception as e:
        fig = _malformed_fig(str(e))
    else:
        # Not the plt.figure() from above, just in case the code creates a new figure.
        fig = plt.gcf()
    return fig


def _make_images(
    filepath: pathlib.Path,
    dirpath: pathlib.Path,
    figcache: Optional[dict],
    last_init_code: str,
    last_init_scope: dict,
) -> Tuple[str, dict]:
    for file in dirpath.iterdir():
        if file.name.endswith(".png"):
            file.unlink()
    with open(filepath) as f:
        contents = f.read()
    inits = list(pyimageinit_re.finditer(contents))
    if len(inits) == 0:
        malformed = False
        init_code = ""
        init_scope = {}
        if last_init_code != "":
            figcache = {}  # also reset our figcache
    elif len(inits) == 1:
        malformed = False
        [init_code] = inits[0].groups()
        init_code = textwrap.dedent(init_code).strip()
        if init_code == last_init_code:
            # skip re-exec'ing
            init_scope = last_init_scope
        else:
            init_scope = {}
            figcache = {}  # also reset our figcache
            try:
                exec(init_code, init_scope)
            except Exception as e:
                malformed = True
                malformed_msg = "In pyimageinit: " + str(e)
    else:
        malformed = True
        malformed_msg = "Cannot have multiple #pyimageinit directives"
        init_code = ""
        init_scope = {}
        figcache = {}  # needed in case we go 1->2->0 number of pyimageinits
    for i, match in enumerate(pyimage_re.finditer(contents)):
        if malformed:
            fig = _malformed_fig(malformed_msg)  # pyright: ignore
        else:
            i = i + 1
            [code] = match.groups()
            code = textwrap.dedent(code).strip()
            if figcache is None:
                fig = _make_fig(code, init_scope)
            else:
                try:
                    fig = figcache[code]
                except KeyError:
                    fig = _make_fig(code, init_scope)
        fig.savefig(dirpath / f"{i}.png")
    return init_code, init_scope


def _get_file_time(filepath: pathlib.Path) -> int:
    return filepath.stat().st_mtime_ns


def _initial(filename: Union[str, pathlib.Path], figcache: Optional[dict]):
    installpath = pathlib.Path(__file__).resolve().parent
    filepath = pathlib.Path(filename).resolve()
    dirpath = filepath.parent / ".typst_pyimage"
    dirpath.mkdir(exist_ok=True)
    shutil.copy(installpath / "pyimage.typ", dirpath / "pyimage.typ")
    init_code, init_scope = _make_images(filepath, dirpath, figcache, "", {})
    return filepath, dirpath, init_code, init_scope


def watch(filename: Union[str, pathlib.Path], timeout_s: Optional[int] = None):
    figcache = {}
    filepath, dirpath, init_code, init_scope = _initial(filename, figcache)
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
                init_code, init_scope = _make_images(
                    filepath, dirpath, figcache, init_code, init_scope
                )
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
    filepath, _, _, _ = _initial(filename, figcache=None)
    subprocess.run(["typst", "compile", str(filepath)])
