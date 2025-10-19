#!/usr/bin/env python3
import os
import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

FOLDER_RE = re.compile(r"checkpoint-(?P<ckpt>\d+)-n\d+_s(?P<shot>\d+)$")
NUM_RE = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")
BASE_METRICS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
CANONICAL_METRICS = BASE_METRICS + [m + "_std" for m in BASE_METRICS]


def _norm(s): return re.sub(r"\s+", "_", s).lower()

NORM_TO_CANON = {}
for m in CANONICAL_METRICS:
    NORM_TO_CANON[_norm(m)] = m
    NORM_TO_CANON[_norm(m).replace("_", " ")] = m

def get_color(shot, max_shot, cmap="tab20"):
    cmap = plt.get_cmap(cmap)
    return cmap(shot / max_shot) if max_shot else cmap(0)

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

def make_dataframe(base_dir):
    rows = []
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        mo = FOLDER_RE.match(name)
        if not mo:
            continue
        parsed = parse_results_log(os.path.join(full, "results.log"))
        if parsed is None:
            continue
        rows.append({"checkpoint": int(mo.group("ckpt")), "shot": int(mo.group("shot")), "folder": name, **parsed})
    if not rows:
        print("ERROR: No results found.", file=sys.stderr); sys.exit(2)
    return pd.DataFrame(rows).sort_values(["shot", "checkpoint"]).reset_index(drop=True)

def plot_lines(metric, df, out, max_shot):
    if metric not in df.columns:
        return
    plt.figure()
    for shot, sub in df.groupby("shot"):
        sub = sub.sort_values("checkpoint")
        plt.plot(sub["checkpoint"], sub[metric], marker="o", label=f"{shot}-shot", color=get_color(shot, max_shot))
    plt.xlabel("checkpoint"); plt.ylabel("accuracy"); plt.title(metric); plt.legend(); plt.grid(True, ls=":"); plt.tight_layout()
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_mean_std_band(metric, df, out, max_shot, window=3):
    if metric not in df.columns:
        return
    plt.figure()
    for shot, sub in df.groupby("shot"):
        agg = sub.groupby("checkpoint")[metric].agg(["mean", "std"]).rename(columns={"mean":"mu","std":"sigma"}).sort_index()
        xs = agg.index.values
        mu = agg["mu"].values
        sigma = agg["sigma"].fillna(0).values
        if window and window > 1:
            mu = pd.Series(mu).rolling(window=window, center=True).mean().to_numpy()
            sigma = pd.Series(sigma).rolling(window=window, center=True).mean().to_numpy()
        c = get_color(shot, max_shot)
        plt.plot(xs, mu, lw=2.5, label=f"{shot}-shot", color=c)
        plt.fill_between(xs, mu - sigma, mu + sigma, alpha=0.2, color=c)
    plt.xlabel("checkpoint"); plt.ylabel("accuracy")
    title = f"{metric}" + (f" (smoothed w={window})" if window and window > 1 else "")
    plt.title(title); plt.legend(); plt.grid(True, ls=":"); plt.tight_layout()
    plt.savefig(out, bbox_inches="tight"); plt.close()

def main():
    p = argparse.ArgumentParser(description="Summarize results and produce plots.")
    p.add_argument("--base_dir", type=str, required=True, help="Directory with checkpoint-* subfolders")
    p.add_argument("--output_dir", type=str, default="plots")
    p.add_argument("--smooth_window_band", type=int, default=3)
    args = p.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"ERROR: Base directory not found: {args.base_dir}", file=sys.stderr); sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    df = make_dataframe(args.base_dir)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    max_shot = int(df["shot"].max()) if not df["shot"].isnull().all() else 0

    for metric in BASE_METRICS:
        base_name = metric.replace(" ", "_")
        out_lines = os.path.join(args.output_dir, f"{base_name}.png")
        out_band = os.path.join(args.output_dir, f"banded_{base_name}.png")
        plot_lines(metric, df, out_lines, max_shot)
        plot_mean_std_band(metric, df, out_band, max_shot, window=args.smooth_window_band)

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(df.head(20))

if __name__ == "__main__":
    main()
