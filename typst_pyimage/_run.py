import dataclasses
import enum
import hashlib
import inspect
import json
import logging
import pathlib
import shutil
import subprocess
import time

import matplotlib
import matplotlib.pyplot as plt


_here = pathlib.Path(__file__).resolve().parent
_logger = logging.getLogger("typst_pyimage")


class _OutputType(enum.Enum):
    content = ".txt"
    image = ".png"


@dataclasses.dataclass(frozen=True)
class _Output:
    raw_text: str
    clean_text: str
    filename: str
    type: _OutputType


@dataclasses.dataclass(frozen=True)
class _Result:
    init_text: str
    namespace: dict


def _new_namespace(filepath: pathlib.Path) -> dict:
    return {"__name__": "<dynamic in typst_pyimage>", "__file__": str(filepath)}


def _build(
    filepath: pathlib.Path,
    old_init_text: str,
    old_namespace: dict,
    log_success: bool,
    force_build: bool,
) -> None | _Result | Exception:
    # Step 1: prepare the `.typst_pyimage` folder so that the Typst code is valid.
    assert filepath.is_absolute()
    outdir = filepath.parent / ".typst_pyimage"
    outdir.mkdir(exist_ok=True)
    here_pyimage = _here / "pyimage.typ"
    out_pyimage = outdir / "pyimage.typ"
    if not out_pyimage.exists() or out_pyimage.read_text() != here_pyimage.read_text():
        # Don't copy unconditionally, to avoid Typst recompiling unnecessarily.
        shutil.copy(_here / "pyimage.typ", out_pyimage)

    # Step 2: Query Typst.
    p = subprocess.run(
        [
            "typst",
            "query",
            str(filepath),
            "metadata",
            "--input",
            "typst_pyimage.query=true",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if p.returncode != 0:
        return RuntimeError(f"stdout: {p.stdout}\nstderr: {p.stderr}")
    if log_success:
        _logger.info("Error resolved.")
    data = json.loads(p.stdout)

    # Step 3: Determine the desired state of the world.

    # Step 3a: get the `pyinit` text.
    init_text: None | str = None
    for info in data:
        assert info["func"] == "metadata"
        value: str = info["value"]
        if value.startswith("typst_pyimage.pyinit."):
            if init_text is None:
                init_text = value.removeprefix("typst_pyimage.pyinit.")
            else:
                return RuntimeError("Multiple `pyinit` statements found.")
    if init_text is None:
        init_text = ""

    # Step 3b: get the output texts, and the output files they will write to.
    outputs: list[_Output] = []
    for info in data:
        value: str = info["value"]
        if value.startswith("typst_pyimage."):
            if value.startswith("typst_pyimage.pyinit."):
                continue
            elif value.startswith("typst_pyimage.pycontent."):
                raw_text = value.removeprefix("typst_pyimage.pycontent.")
                clean_text = raw_text.strip()
                output_type = _OutputType.content
            elif value.startswith("typst_pyimage.pyimage."):
                raw_text = value.removeprefix("typst_pyimage.pyimage.")
                clean_text = inspect.cleandoc(raw_text).strip()
                output_type = _OutputType.image
            else:
                assert False
            h = hashlib.new("sha1")
            h.update(init_text.encode())
            h.update(clean_text.encode())
            outputs.append(
                _Output(
                    raw_text=raw_text,
                    clean_text=clean_text,
                    filename=h.hexdigest() + output_type.value,
                    type=output_type,
                )
            )

    # Step 4: check if we're aleady in that state of the world, and do nothing if so.
    desired_filenames = {output.filename for output in outputs}
    desired_filenames.add("pyimage.typ")
    desired_filenames.add("mapping.json")
    if not force_build and all(
        (outdir / filename).exists() for filename in desired_filenames
    ):
        return None

    # Step 5: we're not in that state of the world. Make it so.

    # Step 5a: delete what we don't need.
    for existing_output_file in outdir.iterdir():
        if existing_output_file.name not in desired_filenames:
            # Tolerant to race condition with being deleted exogenously.
            existing_output_file.unlink(missing_ok=True)

    # Step 5b: run `pyinit` if needed. This is frequently expensive so we try not to do
    # this if possible.
    if init_text == old_init_text:
        namespace = old_namespace
    else:
        _logger.info("Running `pyinit`.")
        namespace = _new_namespace(filepath)
        exec(init_text, namespace)

    # Step 5c: run all `pyimage` and `pycontent` blocks.
    for output in outputs:
        if (outdir / output.filename).exists():
            continue
        _logger.info("Building output %s", output.filename)
        if output.type == _OutputType.content:
            content_value = eval(output.clean_text, namespace.copy())
            (outdir / output.filename).write_text(str(content_value))
        elif output.type == _OutputType.image:
            plt.figure()  # Don't overwrite previous figure.
            with matplotlib.rc_context():
                exec(output.clean_text, namespace.copy())
            # Not the `plt.figure()` above in case the code creates its own figure.
            fig = plt.gcf()
            fig.savefig(outdir / output.filename)
        else:
            assert False

    # Step 5d: built mapping file.
    mapping = {output.raw_text: output.filename for output in outputs}
    (outdir / "mapping.json").write_text(json.dumps(mapping))

    return _Result(init_text, namespace)


_spacer = "\n-----------------------"


def compile(filepath: pathlib.Path, should_raise: bool) -> None:
    out = _build(
        filepath, "", _new_namespace(filepath), log_success=False, force_build=False
    )
    if out is None:
        _logger.info("Nothing to do, already built.")
    elif isinstance(out, Exception):
        if should_raise:
            raise out
        else:
            _logger.warning(str(out))
    elif isinstance(out, _Result):
        pass
    else:
        assert False


def watch(filepath: pathlib.Path, should_raise: bool) -> None:
    init_text = ""
    namespace = _new_namespace(filepath)
    str_warning = None
    log_success = False
    first_build = True
    while True:
        out = _build(
            filepath, init_text, namespace, log_success, force_build=first_build
        )
        if isinstance(out, Exception):
            if should_raise:
                raise out
            else:
                log_success = True
                new_str_warning = str(out).strip()
                if str_warning != new_str_warning:
                    _logger.warning(new_str_warning + _spacer)
                    str_warning = new_str_warning
        else:
            log_success = False
            str_warning = None
            if out is None:
                pass
            elif isinstance(out, _Result):
                init_text = out.init_text
                namespace = out.namespace
                _logger.info("Built successfully." + _spacer)
            else:
                assert False
        first_build = False
        time.sleep(2)
