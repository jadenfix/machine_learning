#!/usr/bin/env python3
"""Fail closed when published media or its recorded source drifts."""

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"


def run(*args, check=True):
    return subprocess.run(
        args, cwd=ROOT, text=True, capture_output=True, check=check
    )


def main():
    receipt = json.loads((DEMO / "capture.json").read_text())
    source = receipt["recorded_source_sha"]
    ancestor = run("git", "merge-base", "--is-ancestor", source, "HEAD", check=False)
    if ancestor.returncode != 0:
        raise SystemExit(f"recorded source {source} is not an ancestor of HEAD")
    tree = run("git", "rev-parse", f"{source}^{{tree}}").stdout.strip()
    if tree != receipt["recorded_source_tree"]:
        raise SystemExit("recorded source tree does not match")
    for path, expected_blob in receipt["source_blobs"].items():
        recorded_blob = run("git", "rev-parse", f"{source}:{path}").stdout.strip()
        head_blob = run("git", "rev-parse", f"HEAD:{path}").stdout.strip()
        if recorded_blob != expected_blob or head_blob != expected_blob:
            raise SystemExit(f"source drift: {path}")
    for name, expected in receipt["artifacts"].items():
        path = DEMO / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"artifact drift: {name}")
        if path.suffix in {".gif", ".mp4"}:
            run("ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-")
            probe = json.loads(
                run(
                    "ffprobe", "-v", "error", "-show_entries",
                    "stream=codec_name,width,height,nb_frames", "-of", "json", str(path)
                ).stdout
            )
            stream = probe["streams"][0]
            if stream["width"] != 960 or stream["height"] != 540:
                raise SystemExit(f"unexpected dimensions: {name}")
            if int(stream.get("nb_frames", 0)) < 150:
                raise SystemExit(f"too few decoded frames: {name}")
    print(
        f"verified {len(receipt['source_blobs'])} source blobs and "
        f"{len(receipt['artifacts'])} artifacts at {source}"
    )


if __name__ == "__main__":
    main()
