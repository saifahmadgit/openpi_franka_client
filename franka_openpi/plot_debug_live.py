#!/usr/bin/env python3
"""
Watch the debug directory for updated .npy files and regenerate the PNG on the fly.

Usage:
    python plot_debug_live.py [--debug_dir ~/Franka/src/franka_openpi/debug] [--interval 1.0]
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(commanded: np.ndarray, actual: np.ndarray, path: str):
    n = min(len(commanded), len(actual))
    if n == 0:
        return
    cmd, act = commanded[:n], actual[:n]
    steps = np.arange(n)

    fig, axes = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(f"Policy Command vs Robot State — {n} steps", fontsize=13)
    for j, ax in enumerate(axes):
        ax.scatter(steps, cmd[:, j], color="steelblue", s=10, label="commanded (policy)")
        ax.scatter(steps, act[:, j], color="tomato",    s=10, label="actual (robot state)")
        ax.set_ylabel(f"Joint {j + 1}\n(rad)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {n} steps → {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir", default=os.path.expanduser("~/Franka/src/franka_openpi/debug"))
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds")
    args = parser.parse_args()

    cmd_path = os.path.join(args.debug_dir, "commanded.npy")
    act_path = os.path.join(args.debug_dir, "actual.npy")
    out_path = os.path.join(args.debug_dir, "act_debug.png")

    last_mtime = 0.0
    print(f"Watching {args.debug_dir} every {args.interval}s — Ctrl+C to stop", flush=True)

    while True:
        try:
            mtime = max(os.path.getmtime(cmd_path), os.path.getmtime(act_path))
        except FileNotFoundError:
            time.sleep(args.interval)
            continue

        if mtime > last_mtime:
            try:
                cmd = np.load(cmd_path)
                act = np.load(act_path)
                plot(cmd, act, out_path)
                last_mtime = mtime
            except Exception as e:
                print(f"[plot] error: {e}", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
