#!/usr/bin/env python3
import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER_RE = re.compile(
    r"checkpoint-(?P<step>\d+)_t(?P<temperature>[-+]?\d*\.?\d+)_n(?P<test_sample_size>\d+)_s(?P<shot>\d+)_seed(?P<seed>\d+)",
    re.IGNORECASE,
)
BASE_METRICS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
CANONICAL_METRICS = BASE_METRICS + [m + "_se" for m in BASE_METRICS]
METRIC_KEYS = {
    re.sub(r"[\s_]+", "", metric).lower(): metric
    for metric in CANONICAL_METRICS
}

def parse_folder_name(name):
    match = FOLDER_RE.fullmatch(name)
    if not match:
        return None
    info = {
        "step": int(match.group("step")),
        "shot": int(match.group("shot")),
        "temperature": float(match.group("temperature")),
        "test_sample_size": int(match.group("test_sample_size")),
        "seed": int(match.group("seed")),
    }
    return info

def get_color(shot, max_shot, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    if max_shot and max_shot > 0:
        norm_shot = 1.0 - (shot / max_shot)
        return cmap(norm_shot)
    else:
        return cmap(0.0)

def parse_results_log(path):
    vals = {m: None for m in BASE_METRICS}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s*([^:]+)\s*:\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if not m:
                continue
            label, num = m.group(1).strip(), m.group(2)
            canon = METRIC_KEYS.get(re.sub(r"[\s_]+", "", label).lower())
            vals[canon] = float(num)
    return vals


def iter_result_rows(base_dir):
    abs_base = os.path.abspath(base_dir)
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        info = parse_folder_name(name)
        if not info:
            continue
        parsed = parse_results_log(os.path.join(full, "results.log"))
        if parsed is None:
            continue
        row = {"source_dir": abs_base, **info}
        for metric in BASE_METRICS:
            row[metric] = parsed.get(metric)
        yield row

def make_dataframe(base_dirs):
    frames = []
    for base_dir in base_dirs:
        rows = list(iter_result_rows(base_dir))
        df = pd.DataFrame(rows).sort_values(["shot", "step"]).reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["source_dir", "shot", "step"]).reset_index(drop=True)

def aggregate_metrics(df):
    grouped = df.groupby(["shot", "step", "temperature", "test_sample_size"], dropna=False)

    summary = grouped[BASE_METRICS].mean()
    summary["seed_count"] = grouped["seed"].nunique()

    std_df = grouped[BASE_METRICS].std(ddof=0).fillna(0)
    for metric in BASE_METRICS:
        summary[f"{metric}_se"] = std_df[metric] / np.sqrt(summary["seed_count"])

    return summary.reset_index().sort_values(["shot", "step"]).reset_index(drop=True)


def plot_lines(metric, df, out, max_shot=10):
    plt.figure()
    se_col = f"{metric}_se"
    for shot, sub in df.groupby("shot", dropna=False):
        sub = sub.sort_values("step")
        color = get_color(shot, max_shot)
        plt.plot(sub["step"], sub[metric], marker="o", label=f"{shot}-shot", color=color)
        if se_col in sub.columns:
            std_vals = sub[se_col].fillna(0).to_numpy()
            if np.any(std_vals > 0):
                lower = sub[metric] - std_vals
                upper = sub[metric] + std_vals
                plt.fill_between(sub["step"], lower, upper, color=color, alpha=0.2)
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title(metric)
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Summarize results and produce plots.")
    p.add_argument("--base_dir", type=str, nargs="+", required=True, help="One or more directories with checkpoint-* subfolders")
    p.add_argument("--output_dir", type=str, default="plots")
    args = p.parse_args()

    raw_df = make_dataframe(args.base_dir)
    agg_df = aggregate_metrics(raw_df)

    os.makedirs(args.output_dir, exist_ok=True)
    raw_df.to_csv(args.output_dir+"/results.csv", index=False)
    agg_df.to_csv(args.output_dir+"/results_summary.csv", index=False)

    for metric in BASE_METRICS:
        plot_lines(metric, agg_df, args.output_dir+f"/{metric}.png")

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\nPreview of aggregated results:")
        print(agg_df.head(20))

if __name__ == "__main__":
    main()
