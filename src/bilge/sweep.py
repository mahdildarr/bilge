"""Training-set size x LoRA rank grid: how much data and capacity does Turkish instruction tuning need?

Every grid point is a full ``bilge.train`` run with alpha = 2 x rank; the resulting perplexities and
training times are collected into one table.

    python -m bilge.sweep --config configs/sweep.yaml --out runs/sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml


def grid_points(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def run_name(point: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in point.items())


def table(rows: list[dict]) -> str:
    lines = ["| Train examples | LoRA rank | Trainable params | Perplexity before | Perplexity after | Train time |",
             "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["train_examples"], r["config"]["rank"])):
        lines.append(f"| {r['train_examples']:,} | {r['config']['rank']} | {r['trainable_parameters']:,} | "
                     f"{r['perplexity_before']} | {r['perplexity_after']} | {r['train_runtime_s'] / 60:.0f} min |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Train-size x rank sweep")
    p.add_argument("--config", default="configs/sweep.yaml")
    p.add_argument("--out", default="runs/sweep")
    a = p.parse_args(argv)
    from .train import main as train_main

    spec = yaml.safe_load(Path(a.config).read_text())
    rows = []
    for point in grid_points(spec["grid"]):
        out = Path(a.out) / run_name(point)
        if not (out / "metrics.json").exists():  # resumable: finished points are skipped
            overrides = [f"{k}={v}" for k, v in point.items()] + [f"out={out}"]
            if "rank" in point:
                overrides.append(f"alpha={2 * point['rank']}")
            train_main(["--config", spec["base"], "--set", *overrides])
        rows.append(json.loads((out / "metrics.json").read_text(encoding="utf-8")))
    text = table(rows)
    (Path(a.out) / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
