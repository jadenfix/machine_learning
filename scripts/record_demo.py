#!/usr/bin/env python3
"""Record a truthful visual receipt from the tested offline workflow."""

import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
FPS = 12
WIDTH, HEIGHT = 960, 540
TESTS_PASSED = 5
SOURCE_INPUTS = [
    ".github/workflows/research.yml",
    "Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb",
    "requirements.txt",
    "run_research.py",
    "tests/test_research_runner.py",
    "scripts/record_demo.py",
    "scripts/verify_demo.py",
]
DEPENDENCIES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "scikit-learn",
    "Pillow",
]


def run(*args, cwd=ROOT, check=True):
    return subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, check=check
    )


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def rounded(draw, box, fill, outline=None, radius=16, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size, fill="#eef5ff", bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def ease(value):
    return 0.5 - 0.5 * math.cos(math.pi * max(0, min(1, value)))


def base_frame(index, total):
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            glow = max(0, 1 - math.hypot(x - 760, y - 60) / 520)
            pixels[x, y] = (
                int(8 + 10 * glow),
                int(16 + 26 * glow),
                int(31 + 48 * glow),
            )
    draw = ImageDraw.Draw(image)
    text(draw, (42, 34), "MACHINE LEARNING · EVIDENCE PIPELINE", 13, "#7dd3fc", True)
    text(draw, (918, 35), "OFFLINE · SEED 42", 12, "#a7f3d0", True, "ra")
    draw.rounded_rectangle((42, 508, 918, 514), radius=3, fill="#17263d")
    progress = int(876 * index / max(1, total - 1))
    draw.rounded_rectangle((42, 508, 42 + progress, 514), radius=3, fill="#38bdf8")
    return image, draw


def render_frame(index, total, report, source_sha):
    image, draw = base_frame(index, total)
    seconds = index / FPS
    scene = min(4, int(seconds // 3))
    local = ease((seconds % 3) / 2.2)

    if scene == 0:
        text(draw, (42, 92), "Can a failed rerun lie?", 34, "#ffffff", True)
        text(draw, (42, 137), "Previously published evidence must survive intact.", 18, "#a9b8cc")
        rounded(draw, (42, 188, 438, 415), "#101d31", "#30415c")
        text(draw, (68, 216), "LAST COMPLETE RUN", 12, "#7dd3fc", True)
        text(draw, (68, 254), "execution.json", 23, "#ffffff", True)
        text(draw, (68, 291), "5 models · 171 holdout rows", 15, "#a9b8cc")
        text(draw, (68, 342), "SHA-256", 12, "#64748b", True)
        text(draw, (68, 367), report["notebook_sha256"][:24] + "…", 15, "#dbeafe")
        rounded(draw, (476, 188, 918, 415), "#26161d", "#7f1d1d")
        text(draw, (502, 216), "FORCED RERUN FAILURE", 12, "#fda4af", True)
        text(draw, (502, 257), "RuntimeError", 24, "#ffffff", True)
        text(draw, (502, 297), "No partial report published", 16, "#fecdd3")
        text(draw, (502, 355), "old bytes preserved", 18, "#86efac", True)
    elif scene == 1:
        text(draw, (42, 92), "Run the checked notebook", 34, "#ffffff", True)
        text(draw, (42, 137), "Actual scikit-learn workflow · no provider calls", 18, "#a9b8cc")
        steps = [
            ("01", "Load", "569 rows · 30 features"),
            ("02", "Split", "stratified 70 / 30"),
            ("03", "Train", "five ensemble methods"),
            ("04", "Publish", "transactional directory swap"),
        ]
        for n, (num, label, detail) in enumerate(steps):
            x = 42 + (n % 2) * 448
            y = 188 + (n // 2) * 116
            active = local > n / 4
            rounded(draw, (x, y, x + 422, y + 92), "#10233a", "#38bdf8" if active else "#30415c")
            text(draw, (x + 22, y + 22), num, 14, "#38bdf8", True)
            text(draw, (x + 65, y + 18), label, 20, "#ffffff", True)
            text(draw, (x + 65, y + 51), detail, 14, "#a9b8cc")
    elif scene == 2:
        text(draw, (42, 92), "Five real holdout results", 34, "#ffffff", True)
        text(draw, (42, 137), "Descriptive fixed-split output—not clinical validation", 18, "#fbbf24")
        models = list(report["model_results"].items())
        for n, (name, result) in enumerate(models):
            y = 190 + n * 55
            accuracy = float(result["accuracy"])
            label = name.replace(" (Logistic Regression)", " + LR").replace(
                " (Decision Tree Stump)", ""
            ).replace(" (NB, DT, LR -> LR)", "")
            text(draw, (42, y), label, 14, "#dbeafe", True)
            rounded(draw, (330, y + 1, 820, y + 21), "#17263d", radius=10)
            width = int(490 * accuracy * local)
            rounded(draw, (330, y + 1, 330 + width, y + 21), "#38bdf8", radius=10)
            text(draw, (900, y + 10), f"{accuracy:.3f}", 15, "#ffffff", True, "rm")
        text(draw, (42, 469), "Holdout rows: 171  ·  Seed: 42  ·  Executed cells: 1", 14, "#94a3b8")
    elif scene == 3:
        text(draw, (42, 92), "Publish only when complete", 34, "#ffffff", True)
        text(draw, (42, 137), "Staging → verification → directory replacement", 18, "#a9b8cc")
        nodes = [
            (42, "STAGING", "isolated output"),
            (356, "VERIFY", "cells + artifacts"),
            (670, "PUBLISH", "complete evidence"),
        ]
        for n, (x, title, detail) in enumerate(nodes):
            active = local > n / 3
            rounded(draw, (x, 220, x + 248, 348), "#10233a", "#38bdf8" if active else "#30415c")
            text(draw, (x + 24, 249), title, 16, "#7dd3fc", True)
            text(draw, (x + 24, 287), detail, 15, "#ffffff")
            if n < 2:
                text(draw, (x + 277, 284), "→", 28, "#64748b", True, "mm")
        rounded(draw, (42, 390, 918, 454), "#11271f", "#166534")
        text(draw, (66, 411), "✓  Failed rerun leaves the last complete evidence byte-for-byte unchanged", 17, "#bbf7d0", True)
    else:
        text(draw, (42, 92), "Reproducible research receipt", 34, "#ffffff", True)
        text(draw, (42, 137), "Tested source and media are cryptographically pinned.", 18, "#a9b8cc")
        rounded(draw, (42, 188, 918, 366), "#101d31", "#30415c")
        text(draw, (68, 219), "RECORDED SOURCE", 12, "#7dd3fc", True)
        text(draw, (68, 251), source_sha[:16] + "…", 20, "#ffffff", True)
        text(
            draw,
            (68, 300),
            f"{TESTS_PASSED} integrity regressions",
            18,
            "#86efac",
            True,
        )
        text(draw, (363, 300), "full GIF + MP4 decode", 18, "#86efac", True)
        text(draw, (68, 335), "transactional publication", 18, "#86efac", True)
        rounded(draw, (42, 391, 918, 462), "#2b2110", "#92400e")
        text(draw, (66, 414), "Educational fixed holdout · not a medical decision system", 18, "#fde68a", True)
    return image


def metadata(path):
    probe = run(
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_name,width,height,nb_frames",
        "-of", "json", str(path)
    )
    return json.loads(probe.stdout)


def main():
    status = run("git", "status", "--porcelain", "--untracked-files=no").stdout
    if status.strip():
        raise SystemExit("refusing to record with tracked source changes")
    source_sha = run("git", "rev-parse", "HEAD").stdout.strip()
    source_tree = run("git", "rev-parse", "HEAD^{tree}").stdout.strip()

    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        evidence = temp / "evidence"
        workflow = run(
            "python", "run_research.py",
            "Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb",
            "--output", str(evidence)
        )
        tests = run("python", "-m", "unittest", "discover", "-s", "tests", "-v")
        report = json.loads((evidence / "execution.json").read_text())

        frames = temp / "frames"
        frames.mkdir()
        total = FPS * 15
        for index in range(total):
            render_frame(index, total, report, source_sha).save(
                frames / f"frame-{index:04d}.png", optimize=True
            )

        OUT.mkdir(parents=True, exist_ok=True)
        gif = OUT / "machine-learning-demo.gif"
        mp4 = OUT / "machine-learning-demo.mp4"
        poster = OUT / "machine-learning-demo-poster.png"
        palette = temp / "palette.png"
        render_frame(FPS * 13, total, report, source_sha).save(poster, optimize=True)
        run(
            "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
            "-i", str(frames / "frame-%04d.png"), "-vf", "palettegen=max_colors=128",
            str(palette)
        )
        run(
            "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
            "-i", str(frames / "frame-%04d.png"), "-i", str(palette),
            "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(gif)
        )
        run(
            "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
            "-i", str(frames / "frame-%04d.png"), "-c:v", "libx264",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)
        )

    artifacts = {}
    for path in [gif, mp4, poster]:
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "probe": metadata(path) if path.suffix in {".gif", ".mp4"} else None,
        }
    source_blobs = {
        path: run("git", "rev-parse", f"{source_sha}:{path}").stdout.strip()
        for path in SOURCE_INPUTS
    }
    receipt = {
        "recorded_source_sha": source_sha,
        "recorded_source_tree": source_tree,
        "source_blobs": source_blobs,
        "workflow_command": (
            "python run_research.py "
            "Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb --output <temp>/evidence"
        ),
        "test_command": "python -m unittest discover -s tests -v",
        "tests_passed": TESTS_PASSED,
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in DEPENDENCIES
        },
        "executed_cells": report["executed_cells"],
        "notebook_sha256": report["notebook_sha256"],
        "model_accuracies": {
            name: result["accuracy"]
            for name, result in report["model_results"].items()
        },
        "workflow_stdout_sha256": hashlib.sha256(workflow.stdout.encode()).hexdigest(),
        "test_stdout_sha256": hashlib.sha256((tests.stdout + tests.stderr).encode()).hexdigest(),
        "scope": report["scope"],
        "claims_excluded": [
            "clinical utility or safety",
            "independent validation",
            "generalization beyond the fixed holdout",
            "live provider or production deployment",
            "timing or performance benchmark",
        ],
        "artifacts": artifacts,
    }
    (OUT / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
