import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_phase(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["phase"]].append(rec)
    return grouped


def plot_phase(records, phase, output_dir):
    if not records:
        return

    keys = [k for k in records[0].keys() if k not in {"phase", "step", "step_ratio"}]
    keys = [k for k in keys if any(float(r.get(k, 0.0)) != 0.0 for r in records)]
    if not keys:
        return

    steps = [r["step"] for r in records]

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, max(3, 2.2 * len(keys))), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        values = [float(r.get(key, 0.0)) for r in records]
        ax.plot(steps, values, linewidth=1.5)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.suptitle(f"Stage2 Loss Curves: {phase}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)

    out_path = os.path.join(output_dir, f"loss_{phase}.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.loss_log)
    grouped = group_by_phase(records)

    for phase, phase_records in grouped.items():
        plot_phase(phase_records, phase, args.output_dir)


if __name__ == "__main__":
    main()
