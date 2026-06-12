"""Dump binned scalar trends from a training run's tfevents.

Usage:
    python scripts/diag_run.py output/set5_physdecouple_v13 [--bins 10] [--filter uw/]

Reads every scalar tag and prints per-bin means across the step range, which
makes monotonic drift (e.g. B_inf creep, l1 deterioration) obvious at a glance.
"""
import argparse

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--filter", default="",
                        help="only show tags containing this substring")
    args = parser.parse_args()

    ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
    ea.Reload()

    tags = sorted(t for t in ea.Tags()["scalars"] if args.filter in t)
    print("TAGS:", tags, "\n")

    for tag in tags:
        ev = ea.Scalars(tag)
        steps = np.array([e.step for e in ev])
        vals = np.array([e.value for e in ev])
        if len(vals) == 0:
            continue
        print(f"--- {tag}  (n={len(vals)}, steps {steps.min()}..{steps.max()})")
        edges = np.linspace(steps.min(), steps.max() + 1, args.bins + 1)
        row = []
        for i in range(args.bins):
            m = (steps >= edges[i]) & (steps < edges[i + 1])
            row.append(f"{vals[m].mean():.4f}" if m.any() else "  -  ")
        print("   ", " | ".join(row))


if __name__ == "__main__":
    main()
