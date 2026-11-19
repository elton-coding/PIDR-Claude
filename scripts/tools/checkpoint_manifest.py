"""Validate and maintain ``checkpoints/MANIFEST.csv``.

The manifest is the single record of which weight file belongs to which dataset,
experiment and paper result. Layout: ``checkpoints/<dataset>/<experiment>/[<setting>/]<stage>.pth``
with ``stage`` in {diffusion, refiner, pinn}; unused weights live in ``checkpoints/archive/``.

Usage (repo root):
    python -m scripts.tools.checkpoint_manifest check    # verify files, md5 and architecture
    python -m scripts.tools.checkpoint_manifest update   # refresh computed columns, append untracked files

Hand-written columns (``paper_usage``, ``evidence``) are never changed by ``update``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
import sys
from typing import Dict, List

import torch

ROOT = "checkpoints"
MANIFEST = os.path.join(ROOT, "MANIFEST.csv")
FIELDS = [
    "path", "dataset", "experiment", "setting", "stage", "base_channels",
    "original_name", "trained", "md5", "paper_usage", "evidence",
]


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: str) -> Dict[str, str]:
    """Return the network type and base channel width stored in a state_dict."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if "input_proj.weight" in sd:
        return {"arch": "diffusion", "base_channels": str(sd["input_proj.weight"].shape[0])}
    if "enc.0.block.0.weight" in sd:
        return {"arch": "refiner", "base_channels": str(sd["enc.0.block.0.weight"].shape[0])}
    return {"arch": "unknown", "base_channels": ""}


def split_path(rel: str) -> Dict[str, str]:
    """Derive dataset / experiment / setting / stage from the layout."""
    parts = rel.replace("\\", "/").split("/")
    if parts[0] == "archive":
        return {"dataset": "", "experiment": "archive", "setting": "", "stage": ""}
    stage = os.path.splitext(parts[-1])[0]
    return {
        "dataset": parts[0],
        "experiment": "/".join(parts[1:-2] if len(parts) > 3 else parts[1:-1]),
        "setting": parts[-2] if len(parts) > 3 else "",
        "stage": stage,
    }


def read_manifest() -> List[Dict[str, str]]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: List[Dict[str, str]]) -> None:
    rows.sort(key=lambda r: (r["path"].startswith("archive/"), r["path"]))
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def all_weight_files() -> List[str]:
    found = []
    for base, _, files in os.walk(ROOT):
        for name in files:
            if name.endswith((".pth", ".pt")):
                found.append(os.path.relpath(os.path.join(base, name), ROOT).replace("\\", "/"))
    return sorted(found)


def computed_columns(rel: str) -> Dict[str, str]:
    full = os.path.join(ROOT, rel)
    info = inspect(full)
    cols = split_path(rel)
    cols["base_channels"] = info["base_channels"]
    cols["md5"] = md5sum(full)
    cols["trained"] = datetime.datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
    cols["_arch"] = info["arch"]
    return cols


def check() -> int:
    rows = read_manifest()
    listed = {r["path"] for r in rows}
    problems = 0
    for row in rows:
        full = os.path.join(ROOT, row["path"])
        if not os.path.exists(full):
            print(f"MISSING    {row['path']}")
            problems += 1
            continue
        cols = computed_columns(row["path"])
        if cols["md5"] != row["md5"]:
            print(f"CHANGED    {row['path']} (md5 differs from manifest: file was overwritten)")
            problems += 1
        expected_arch = {"diffusion": "diffusion", "refiner": "refiner", "pinn": "refiner"}.get(row["stage"])
        if expected_arch and cols["_arch"] != expected_arch:
            print(f"WRONG-TYPE {row['path']} (stage={row['stage']} but file is a {cols['_arch']} network)")
            problems += 1
        if row["base_channels"] and cols["base_channels"] != row["base_channels"]:
            print(f"WRONG-CH   {row['path']} (manifest {row['base_channels']}, file {cols['base_channels']})")
            problems += 1
    for rel in all_weight_files():
        if rel not in listed:
            print(f"UNTRACKED  {rel} (run `update`, then fill in paper_usage/evidence)")
            problems += 1
    print(f"{len(rows)} entries checked, {problems} problem(s).")
    return 1 if problems else 0


def update() -> int:
    rows = {r["path"]: r for r in read_manifest()}
    present = set(all_weight_files())
    for rel in sorted(present):
        row = rows.setdefault(rel, {"path": rel, "original_name": os.path.basename(rel)})
        cols = computed_columns(rel)
        cols.pop("_arch")
        row.update(cols)
    for rel in [p for p in rows if p not in present]:
        print(f"dropping missing entry {rel}")
        del rows[rel]
    write_manifest(list(rows.values()))
    print(f"wrote {MANIFEST} ({len(rows)} entries)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "update"])
    args = parser.parse_args()
    return check() if args.command == "check" else update()


if __name__ == "__main__":
    sys.exit(main())
