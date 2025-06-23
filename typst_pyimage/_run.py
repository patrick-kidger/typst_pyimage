import dataclasses
import enum
import hashlib
import inspect
import json
import pathlib
import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt


_here = pathlib.Path(__file__).resolve().parent


class _OutputType(enum.Enum):
    content = ".txt"
    image = ".png"


@dataclasses.dataclass(frozen=True)
class _Output:
    raw_text: str
    clean_text: str
    filename: str
    type: _OutputType


def run(filepath: pathlib.Path) -> None:
    if not filepath.is_absolute():
        raise RuntimeError("Must be provided with an absolute path to the file.")
    outdir = filepath.parent / ".typst_pyimage"
    outdir.mkdir(exist_ok=True)
    shutil.copy(_here / "pyimage.typ", outdir / "pyimage.typ")
    mapping_file = outdir / "mapping.json"
    if not mapping_file.exists():
        mapping_file.write_text("{}")

    p = subprocess.run(
        ["typst", "query", str(filepath), "metadata"],
        capture_output=True,
        check=False,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"stdout: {p.stdout}\nstderr: {p.stderr}")
    data = json.loads(p.stdout)

    # Get the `pyinit` text.
    init_text: None | str = None
    for info in data:
        assert info["func"] == "metadata"
        value: str = info["value"]
        if value.startswith("typst_pyimage.pyinit."):
            if init_text is None:
                init_text = value.removeprefix("typst_pyimage.pyinit.")
            else:
                raise RuntimeError("Multiple `pyinit` statements found.")
    if init_text is None:
        init_text = ""

    # Get the output texts, and the output files they will write to.
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

    # Check if we need to run `pyinit` (or indeed anything at all).
    all_found = True
    for output in outputs:
        if not (outdir / output.filename).exists():
            all_found = False
            break
    if all_found:
        return

    for existing_output_file in outdir.iterdir():
        if existing_output_file != "pyimage.typ":
            existing_output_file.unlink()

    # Run `pyinit`.
    namespace = {}
    exec(init_text, namespace)

    # Run all outputs that need running.
    for output in outputs:
        if (outdir / output.filename).exists():
            continue
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
    mapping = {output.raw_text: output.filename for output in outputs}
    mapping_file.write_text(json.dumps(mapping))
