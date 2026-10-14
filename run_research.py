"""Execute original notebook code without requiring a notebook server.

Results are assembled in a sibling staging directory and published only after every
cell and artifact write succeeds. A failed rerun therefore cannot leave a mixture
of fresh and stale evidence in the requested output directory.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

parser = argparse.ArgumentParser()
parser.add_argument(
    "notebook", choices=["Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb"]
)
parser.add_argument("--output", default="research-output")
args = parser.parse_args()

root = Path(__file__).resolve().parent
os.chdir(root)
out = Path(args.output)
if not out.is_absolute():
    out = root / out
out = out.resolve(strict=False)
if out == root or out in root.parents:
    parser.error("--output must not be the repository root or one of its ancestors")
if out.exists() and not out.is_dir():
    parser.error("--output must be a directory")
# Existing source directories are never an evidence destination.
try:
    relative = out.relative_to(root)
except ValueError:
    relative = None
if relative is not None and relative.parts and relative.parts[0] in {
    ".git", "tests", "scripts", "docs", "archive", "python", "cpp", "java"
}:
    parser.error("--output must not replace a source directory")
out.parent.mkdir(parents=True, exist_ok=True)
stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))

namespace = {"__name__": "__main__"}
import numpy as np

np.random.seed(42)
namespace["display"] = lambda *items: [print(item) for item in items]
started = time.perf_counter()
executed = []

try:
    notebook_path = root / args.notebook
    notebook_bytes = notebook_path.read_bytes()
    for i, cell in enumerate(json.loads(notebook_bytes)["cells"]):
        if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
            continue
        source = "".join(cell["source"])
        if any(
            line.lstrip().startswith(("%", "!")) for line in source.splitlines()
        ):
            raise RuntimeError(
                "Notebook magic requires interactive Jupyter; "
                "this lane must contain ordinary Python"
            )
        print(f"Executing {args.notebook}: cell {i}", flush=True)
        exec(compile(source, f"{args.notebook}:cell{i}", "exec"), namespace)
        executed.append(i)

    import matplotlib.pyplot as plt

    for i in plt.get_fignums():
        plt.figure(i).savefig(
            stage / f"figure-{i}.png", dpi=130, bbox_inches="tight"
        )
    report = {
        "notebook": args.notebook,
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "executed_cells": executed,
        "elapsed_seconds": time.perf_counter() - started,
        "seed": 42,
        "publication": "transactional",
        "sample_controls": {
            key: os.environ[key]
            for key in ["RESEARCH_N", "RESEARCH_MAX_ROWS"]
            if key in os.environ
        },
        "scope": (
            "Five ensemble models train on the built-in Wisconsin Diagnostic "
            "Breast Cancer dataset, with a stratified 70/30 split. This is an "
            "educational fixed holdout evaluation, not a medical decision system "
            "or independent validation study."
        ),
    }
    if "results" in namespace and isinstance(namespace["results"], dict):
        report["model_results"] = namespace["results"]
    (stage / "execution.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )

    previous = None
    if out.exists():
        previous = Path(
            tempfile.mkdtemp(prefix=f".{out.name}.previous-", dir=out.parent)
        )
        previous.rmdir()
        os.replace(out, previous)
    try:
        os.replace(stage, out)
    except BaseException:
        if previous is not None and previous.exists() and not out.exists():
            os.replace(previous, out)
        raise
    if previous is not None:
        shutil.rmtree(previous)
    print(json.dumps(report, indent=2, default=str))
except BaseException:
    shutil.rmtree(stage, ignore_errors=True)
    raise
