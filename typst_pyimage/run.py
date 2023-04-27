import pathlib
import re
import shutil
import subprocess
import textwrap
import time
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt


regex = re.compile(r"pyimage\(\s*\"([^\"]*)\"")


def _make_images(
    filepath: pathlib.Path, dirpath: pathlib.Path, figcache: Optional[dict]
) -> None:
    with open(filepath) as f:
        contents = f.read()
    for i, match in enumerate(regex.finditer(contents)):
        i = i + 1
        [code] = match.groups()
        code = textwrap.dedent(code).strip()
        try:
            if figcache is None:
                raise KeyError
            else:
                fig = figcache[code]
        except KeyError:
            plt.figure()
            try:
                with matplotlib.rc_context():
                    exec(code, {}, {})
            except Exception:
                fig, ax = plt.subplots()
                ax.text(
                    0.5,
                    0.5,
                    "Malformed",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                fig = plt.gcf()
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
