#!/usr/bin/env python3
import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CKPT_RE = re.compile(r"checkpoint-(?P<ckpt>\d+)")
SHOT_RE = re.compile(r"(?:^|[-_])s(?P<shot>\d+)(?:[-_]|$)")
NUM_RE = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")
BASE_METRICS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
CANONICAL_METRICS = BASE_METRICS + [m + "_std" for m in BASE_METRICS]


def _norm(s):
    return re.sub(r"\s+", "_", s).lower()

NORM_TO_CANON = {}
for m in CANONICAL_METRICS:
    NORM_TO_CANON[_norm(m)] = m
    NORM_TO_CANON[_norm(m).replace("_", " ")] = m

def parse_folder_name(name):
    ckpt_match = CKPT_RE.search(name)
    shot_match = SHOT_RE.search(name)
    if not ckpt_match or not shot_match:
        return None
    return int(ckpt_match.group("ckpt")), int(shot_match.group("shot"))

def get_color(shot, max_shot, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    if max_shot and max_shot > 0:
        norm_shot = 1.0 - (shot / max_shot)
        return cmap(norm_shot)
    else:
        return cmap(0.0)

def parse_results_log(path):
    if not os.path.isfile(path):
        return None
    vals = {m: None for m in CANONICAL_METRICS}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s*([^:]+)\s*:\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if not m:
                continue
            label, num = m.group(1).strip(), m.group(2)
            canon = NORM_TO_CANON.get(_norm(label))
            if canon:
                try:
                    vals[canon] = float(num)
                except ValueError:
                    vals[canon] = None
    return None if all(v is None for v in vals.values()) else vals

def make_dataframe(base_dirs):
    frames = []
    for base_dir in base_dirs:
        rows = []
        for name in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            parsed_name = parse_folder_name(name)
            if not parsed_name:
                continue
            ckpt, shot = parsed_name
            parsed = parse_results_log(os.path.join(full, "results.log"))
            if parsed is None:
                continue
            row = {
                "checkpoint": ckpt,
                "shot": shot,
            }
            for metric in BASE_METRICS:
                row[metric] = parsed.get(metric)
            rows.append(row)
        if not rows:
            print(f"WARNING: No results found under {base_dir}.", file=sys.stderr)
            continue
        df = pd.DataFrame(rows).sort_values(["shot", "checkpoint"]).reset_index(drop=True)
        df.insert(0, "source_dir", os.path.abspath(base_dir))
        frames.append(df)
    if not frames:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(2)
    return pd.concat(frames, ignore_index=True).sort_values(["source_dir", "shot", "checkpoint"]).reset_index(drop=True)

def aggregate_metrics(df):
    metrics = [m for m in BASE_METRICS if m in df.columns]
    if not metrics:
        print("ERROR: No metrics found to aggregate.", file=sys.stderr)
        sys.exit(3)
    grouped = df.groupby(["shot", "checkpoint"], dropna=False)
    mean_df = grouped[metrics].mean().reset_index()
    std_df = grouped[metrics].std(ddof=0).fillna(0).reset_index()
    seed_counts = grouped["source_dir"].nunique().reset_index(name="seed_count")
    agg_df = mean_df.merge(seed_counts, on=["shot", "checkpoint"], how="left")
    agg_df = agg_df.set_index(["shot", "checkpoint"])
    std_df = std_df.set_index(["shot", "checkpoint"])
    for metric in metrics:
        agg_df[f"{metric}_seed_std"] = std_df[metric].reindex(agg_df.index).fillna(0)
    return agg_df.reset_index(), metrics


def plot_lines(metric, df, out, max_shot):
    if metric not in df.columns:
        return
    plt.figure()
    std_col = f"{metric}_seed_std"
    for shot, sub in df.groupby("shot", dropna=False):
        sub = sub.sort_values("checkpoint")
        color = get_color(shot, max_shot)
        label = f"{shot}-shot"
        if "seed_count" in sub.columns:
            counts = sub["seed_count"].dropna().unique()
            if len(counts) == 1:
                label += f" (n={int(counts[0])})"
            elif len(counts) > 1:
                label += f" (n={int(counts.min())}-{int(counts.max())})"
        plt.plot(
            sub["checkpoint"], sub[metric],
            marker="o",
            label=label,
            color=color
        )
        if std_col in sub.columns:
            std_vals = sub[std_col].fillna(0).to_numpy()
            if np.any(std_vals > 0):
                lower = sub[metric] - std_vals
                upper = sub[metric] + std_vals
                plt.fill_between(sub["checkpoint"], lower, upper, color=color, alpha=0.2)
    plt.xlabel("checkpoint")
    plt.ylabel("accuracy")
    plt.title(metric)
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved line plot: {out}")


def main():
    p = argparse.ArgumentParser(description="Summarize results and produce plots.")
    p.add_argument("--base_dir", type=str, nargs="+", required=True, help="One or more directories with checkpoint-* subfolders")
    p.add_argument("--output_dir", type=str, default="plots")
    args = p.parse_args()

    base_dirs = []
    for candidate in args.base_dir:
        if not os.path.isdir(candidate):
            print(f"ERROR: Base directory not found: {candidate}", file=sys.stderr)
            sys.exit(1)
        base_dirs.append(candidate)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_df = make_dataframe(base_dirs)
    agg_df, metrics = aggregate_metrics(raw_df)

    raw_csv_path = os.path.join(args.output_dir, "results.csv")
    agg_csv_path = os.path.join(args.output_dir, "results_summary.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"Saved raw results CSV: {raw_csv_path}")
    print(f"Saved aggregated summary CSV: {agg_csv_path}")

    max_shot = int(agg_df["shot"].max()) if not agg_df["shot"].isnull().all() else 0

    for metric in metrics:
        base_name = metric.replace(" ", "_")
        out_lines = os.path.join(args.output_dir, f"{base_name}.png")
        plot_lines(metric, agg_df, out_lines, max_shot)

    print(f"\nAll plots and CSV have been saved to: {os.path.abspath(args.output_dir)}")

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\nPreview of aggregated results:")
        print(agg_df.head(20))

if __name__ == "__main__":
    main()
