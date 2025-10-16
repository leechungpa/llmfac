#!/usr/bin/env python3
"""
Summarize results: per-shot lines + mean±std band plots.

Reads `results.log` files under checkpoint folders like:
  checkpoint-<CKPT>-n<...>_s<SHOT>

Outputs go to a user-specified directory (--out_dir), containing:
- CSV summary
- *_lines.png : original line plot (first version)
- *_band.png  : mean±std band plot (updated version)
"""

import os
import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

FOLDER_RE = re.compile(r"checkpoint-(?P<ckpt>\d+)-n\d+_s(?P<shot>\d+)$")
METRICS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
COLOR_MAP = {0: "C0", 5: "C1", 10: "C2"}

def extract_score(line: str):
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", line)
    return float(m.group(1)) if m else None

def parse_results_log(path: str):
    metrics = {m: None for m in METRICS}
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Average:"):
                metrics["Average"] = extract_score(line)
            elif line.startswith("STEM:"):
                metrics["STEM"] = extract_score(line)
            elif line.startswith("Social Sciences:"):
                metrics["Social Sciences"] = extract_score(line)
            elif line.startswith("Humanities:"):
                metrics["Humanities"] = extract_score(line)
            elif line.startswith("Other:"):
                metrics["Other"] = extract_score(line)
    if all(v is None for v in metrics.values()):
        return None
    return metrics

def make_dataframe(base_dir: str) -> pd.DataFrame:
    rows = []

    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        m = FOLDER_RE.match(name)
        if not m:
            continue
        ckpt = int(m.group("ckpt"))
        shot = int(m.group("shot"))
        log_path = os.path.join(full, "results.log")
        parsed = parse_results_log(log_path)
        if parsed is None:
            continue
        row = {"checkpoint": ckpt, "shot": shot, "folder": name}
        row.update(parsed)
        rows.append(row)
    if not rows:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(2)
    df = pd.DataFrame(rows).sort_values(["shot", "checkpoint"]).reset_index(drop=True)
    return df

def plot_lines(metric_name: str, df: pd.DataFrame, out_path: str):
    plt.figure()
    for shot, sub in df.groupby("shot"):
        sub = sub.sort_values("checkpoint")
        plt.plot(
            sub["checkpoint"],
            sub[metric_name],
            marker="o",
            label=f"{shot}-shot",
            color=COLOR_MAP.get(int(shot), None),
        )
    plt.xlabel("checkpoint")
    plt.ylabel("accuracy")
    plt.title(metric_name)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_mean_std_band(metric_name: str, df: pd.DataFrame, out_path: str, window: int = 5):
    plt.figure()
    for shot, sub in df.groupby("shot"):
        agg = (
            sub.groupby("checkpoint")[metric_name]
            .agg(["mean", "std"])
            .sort_index()
            .rename(columns={"mean": "mu", "std": "sigma"})
        )
        xs = agg.index.values
        mu = agg["mu"].values
        sigma = agg["sigma"].fillna(0.0).values
        if window and window > 1:
            mu = pd.Series(mu).rolling(window=window, center=True).mean().to_numpy()
            sigma = pd.Series(sigma).rolling(window=window, center=True).mean().to_numpy()
        c = COLOR_MAP.get(int(shot), None)
        plt.plot(xs, mu, lw=2.5, label=f"{shot}-shot", color=c)
        plt.fill_between(xs, mu - sigma, mu + sigma, alpha=0.2, color=c)
    plt.xlabel("checkpoint")
    plt.ylabel("accuracy")
    title = f"{metric_name}" + (f" (smoothed w={window})" if window and window > 1 else "")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Summarize Qwen results and produce two plot types.")
    p.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint-* subfolders with results.log files",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="plots",
    )
    p.add_argument("--smooth_window_band", type=int, default=5)
    args = p.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(base_dir):
        print(f"ERROR: Base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    df = make_dataframe(base_dir)
    csv_path = os.path.join(out_dir, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved table to: {csv_path}")

    for metric in METRICS:
        base_name = metric.replace(" ", "_").lower()
        out_lines = os.path.join(out_dir, f"{base_name}.png")
        out_band = os.path.join(out_dir, f"banded_{base_name}.png")
        plot_lines(metric, df, out_lines)
        plot_mean_std_band(metric, df, out_band, window=args.smooth_window_band)
        print(f"Saved: {out_lines}, {out_band}")

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(df.head(10))

if __name__ == "__main__":
    main()
