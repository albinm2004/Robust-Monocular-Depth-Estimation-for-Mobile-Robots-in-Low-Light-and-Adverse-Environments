"""
Aggregate one or more run_eval.py output CSVs into a clean results summary:
mean Abs Rel / RMSE / delta1 (+ near-field variants) broken down by corruption
type, severity, enhancement method, and model size -- formatted as a markdown
table that drops straight into an IEEE paper's Results section, plus the
underlying aggregated CSV for any further slicing.

Usage:
    python scripts/make_results_summary.py data/results/eval_full654_small.csv \
        data/results/eval_subset80_base.csv --out data/results/results_summary.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


METRICS = ["abs_rel", "rmse", "delta1"]
NEAR_FIELD_METRICS = [
    "abs_rel_near_field_0.25_0.70m",
    "rmse_near_field_0.25_0.70m",
    "delta1_near_field_0.25_0.70m",
]


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def aggregate(rows: list[dict]) -> dict[tuple, dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model_size"], r["corruption"], r["severity"], r["enhancement"])
        for m in METRICS + NEAR_FIELD_METRICS:
            v = r.get(m, "")
            if v not in ("", "nan"):
                groups[key][m].append(float(v))
        groups[key]["near_field_pixel_count"].append(int(r.get("near_field_pixel_count", 0)))

    summary = {}
    for key, metric_lists in groups.items():
        summary[key] = {
            "n": len(metric_lists["abs_rel"]),
            **{m: (sum(v) / len(v) if v else float("nan")) for m, v in metric_lists.items()
               if m != "near_field_pixel_count"},
            "near_field_pixel_count_total": sum(metric_lists["near_field_pixel_count"]),
        }
    return summary


def to_markdown(summary: dict[tuple, dict]) -> str:
    lines = [
        "| Model | Corruption | Severity | Enhancement | n | Abs Rel | RMSE (m) | delta1 | "
        "Near-field Abs Rel | Near-field RMSE (m) | Near-field delta1 | Near-field px |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def sort_key(k):
        model, corr, sev, enh = k
        return (model, corr != "clean", corr, int(sev), enh)

    for key in sorted(summary, key=sort_key):
        model, corr, sev, enh = key
        s = summary[key]

        def fmt(v):
            return "-" if v != v else f"{v:.4f}"  # v != v checks NaN

        nf_px = s["near_field_pixel_count_total"]
        nf_cols = (
            [fmt(s.get(m, float("nan"))) for m in NEAR_FIELD_METRICS] if nf_px > 0
            else ["n/a", "n/a", "n/a"]
        )
        lines.append(
            f"| {model} | {corr} | {sev} | {enh} | {s['n']} | "
            f"{fmt(s['abs_rel'])} | {fmt(s['rmse'])} | {fmt(s['delta1'])} | "
            f"{nf_cols[0]} | {nf_cols[1]} | {nf_cols[2]} | {nf_px} |"
        )
    return "\n".join(lines)


def to_csv(summary: dict[tuple, dict], path: str) -> None:
    fieldnames = ["model_size", "corruption", "severity", "enhancement", "n"] + \
        METRICS + NEAR_FIELD_METRICS + ["near_field_pixel_count_total"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, corr, sev, enh), s in summary.items():
            writer.writerow({
                "model_size": model, "corruption": corr, "severity": sev, "enhancement": enh,
                **s,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="One or more run_eval.py output CSVs to aggregate.")
    ap.add_argument("--out", default="data/results/results_summary.md")
    args = ap.parse_args()

    rows = load_rows(args.csvs)
    if not rows:
        print("No rows loaded -- check input CSV paths.")
        return

    summary = aggregate(rows)
    md = to_markdown(summary)

    out_csv = args.out.replace(".md", ".csv")
    to_csv(summary, out_csv)

    header = (
        f"# Results Summary\n\n"
        f"Aggregated from {len(rows)} rows across {len(args.csvs)} CSV(s): "
        f"{', '.join(args.csvs)}\n\n"
        f"Near-field columns use the RealSense D435i's configured obstacle-detection "
        f"band (0.25-0.70m); 'n/a' means this condition has zero ground-truth pixels "
        f"in that band in the underlying data (see README known-simplifications).\n\n"
    )
    with open(args.out, "w") as f:
        f.write(header + md + "\n")

    print(f"Wrote {args.out} and {out_csv} ({len(summary)} groups from {len(rows)} rows)")


if __name__ == "__main__":
    main()
