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
SEED_RE = re.compile(r"(?:^|[-_])seed(?P<seed>\d+)(?:[-_]|$)")
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
    seed_match = SEED_RE.search(name)
    seed = int(seed_match.group("seed")) if seed_match else 0
    return int(ckpt_match.group("ckpt")), int(shot_match.group("shot")), seed


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
    rows = []
    for base_dir in base_dirs:
        base_dir_abs = os.path.abspath(base_dir)
        for name in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            parsed_name = parse_folder_name(name)
            if not parsed_name:
                continue
            ckpt, shot, seed = parsed_name
            parsed = parse_results_log(os.path.join(full, "results.log"))
            if parsed is None:
                continue
            rows.append({
                "source": base_dir_abs,
                "checkpoint": ckpt,
                "shot": shot,
                "seed": seed,
                **parsed
            })
    if not rows:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(2)
    return pd.DataFrame(rows).sort_values(["shot", "checkpoint", "source", "seed"]).reset_index(drop=True)


def plot_lines(metric, df, out, max_shot):
    if metric not in df.columns:
        return
    plt.figure()
    for shot, sub in df.groupby("shot"):
        sub = sub.sort_values("checkpoint")
        run_keys = sub[["seed", "source"]].drop_duplicates()
        num_runs = len(run_keys)
        color = get_color(shot, max_shot)
        if num_runs <= 1:
            label = f"{shot}-shot"
            plt.plot(
                sub["checkpoint"], sub[metric],
                marker="o",
                label=label,
                color=color
            )
        else:
            stats = sub.groupby("checkpoint")[metric].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sigma"}).sort_index()
            xs = stats.index.values
            mu = stats["mu"].values
            sigma = stats["sigma"].fillna(0).values
            plt.plot(xs, mu, marker="o", label=f"{shot}-shot (mean)", color=color)
            if np.any(sigma > 0):
                plt.fill_between(xs, mu - sigma, mu + sigma, alpha=0.25, color=color)
    plt.xlabel("checkpoint")
    plt.ylabel("accuracy")
    plt.title(metric)
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved line plot: {out}")


def plot_mean_std_band(metric, df, out, max_shot, window=3):
    if metric not in df.columns:
        return
    plt.figure()
    for shot, sub in df.groupby("shot"):
        run_keys = sub[["seed", "source"]].drop_duplicates()
        has_multi_seed = len(run_keys) > 1
        agg = sub.groupby("checkpoint")[metric].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sigma"}).sort_index()
        xs = agg.index.values
        mu = agg["mu"].values
        sigma = agg["sigma"].fillna(0).values
        if window and window > 1:
            mu = pd.Series(mu).rolling(window=window, center=True).mean().to_numpy()
            sigma = pd.Series(sigma).rolling(window=window, center=True).mean().to_numpy()
        c = get_color(shot, max_shot)
        label = f"{shot}-shot (mean)" if has_multi_seed else f"{shot}-shot"
        plt.plot(xs, mu, lw=2.5, label=label, color=c)
        if has_multi_seed and np.any(sigma > 0):
            plt.fill_between(xs, mu - sigma, mu + sigma, alpha=0.25, color=c)
    plt.xlabel("checkpoint")
    plt.ylabel("accuracy")
    title = f"{metric}" + (f" (smoothed w={window})" if window and window > 1 else "")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved banded plot: {out}")


def main():
    p = argparse.ArgumentParser(description="Summarize results and produce plots.")
    p.add_argument("--base_dir", type=str, nargs="+", required=True, help="One or more directories with checkpoint-* subfolders")
    p.add_argument("--output_dir", type=str, default="plots")
    p.add_argument("--smooth_window_band", type=int, default=3)
    args = p.parse_args()

    missing = [path for path in args.base_dir if not os.path.isdir(path)]
    if missing:
        print("ERROR: Base directory not found:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    df = make_dataframe(args.base_dir)
    csv_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")

    max_shot = int(df["shot"].max()) if not df["shot"].isnull().all() else 0

    for metric in BASE_METRICS:
        base_name = metric.replace(" ", "_")
        out_lines = os.path.join(args.output_dir, f"{base_name}.png")
        out_band = os.path.join(args.output_dir, f"banded_{base_name}.png")
        plot_lines(metric, df, out_lines, max_shot)
        plot_mean_std_band(metric, df, out_band, max_shot, window=args.smooth_window_band)

    print(f"\nAll plots and CSV have been saved to: {os.path.abspath(args.output_dir)}")

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\nPreview of results:")
        print(df.head(20))

if __name__ == "__main__":
    main()
