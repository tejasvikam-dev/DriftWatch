"""
drift_detector.py
Detects distribution drift between two CSV datasets.
Outputs a single HTML report.

Usage:
  python drift_detector.py --ref reference.csv --curr current.csv
  python drift_detector.py --ref reference.csv --curr current.csv --output my_report.html
"""

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


def ks_test(ref, curr):
    stat, p = stats.ks_2samp(ref.dropna(), curr.dropna())
    return {"stat": round(float(stat), 4), "p": round(float(p), 4), "drifted": p < 0.05}


def chi_squared_test(ref, curr):
    cats = set(ref.dropna().unique()) | set(curr.dropna().unique())
    ref_counts = ref.value_counts().reindex(cats, fill_value=0)
    curr_counts = curr.value_counts().reindex(cats, fill_value=0)
    expected = (ref_counts / ref_counts.sum()) * curr_counts.sum()
    mask = expected > 0
    stat, p = stats.chisquare(f_obs=curr_counts[mask], f_exp=expected[mask])
    return {"stat": round(float(stat), 4), "p": round(float(p), 4), "drifted": p < 0.05}


def psi(ref, curr, bins=10):
    breaks = np.linspace(min(ref.min(), curr.min()), max(ref.max(), curr.max()), bins + 1)
    r = np.clip(np.histogram(ref.dropna(), bins=breaks)[0] / len(ref.dropna()), 1e-6, None)
    c = np.clip(np.histogram(curr.dropna(), bins=breaks)[0] / len(curr.dropna()), 1e-6, None)
    value = float(np.sum((c - r) * np.log(c / r)))
    severity = "none" if value < 0.1 else "moderate" if value < 0.2 else "significant"
    return {"value": round(value, 4), "severity": severity, "drifted": value >= 0.1}


def analyze(ref_df, curr_df):
    results = {}
    for col in [c for c in ref_df.columns if c in curr_df.columns]:
        r, c = ref_df[col], curr_df[col]
        if pd.api.types.is_numeric_dtype(r):
            ks = ks_test(r, c)
            ps = psi(r, c)
            results[col] = {
                "type": "numerical",
                "drifted": ks["drifted"] or ps["drifted"],
                "ref_mean": round(float(r.mean()), 3),
                "curr_mean": round(float(c.mean()), 3),
                "ref_std": round(float(r.std()), 3),
                "curr_std": round(float(c.std()), 3),
                "ks": ks,
                "psi": ps,
            }
        elif pd.api.types.is_object_dtype(r) or isinstance(r.dtype, pd.CategoricalDtype):
            chi = chi_squared_test(r, c)
            results[col] = {
                "type": "categorical",
                "drifted": chi["drifted"],
                "ref_top": r.value_counts(normalize=True).head(4).round(3).to_dict(),
                "curr_top": c.value_counts(normalize=True).head(4).round(3).to_dict(),
                "chi2": chi,
            }
    return results


def build_html(results, ref_path, curr_path, out_path):
    drifted = sum(1 for r in results.values() if r["drifted"])
    stable = len(results) - drifted
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards = ""
    for col, r in results.items():
        color = "#ef4444" if r["drifted"] else "#22c55e"
        label = "DRIFT" if r["drifted"] else "OK"

        if r["type"] == "numerical":
            ks, ps = r["ks"], r["psi"]
            detail = f"""
            <div class="row"><span>Mean</span><span>{r['ref_mean']}</span><span>{r['curr_mean']}</span></div>
            <div class="row"><span>Std</span><span>{r['ref_std']}</span><span>{r['curr_std']}</span></div>
            <div class="row tests"><span>KS test</span><span colspan="2">stat={ks['stat']} &nbsp; p={ks['p']} &nbsp; {'⚠ drifted' if ks['drifted'] else '✓ ok'}</span></div>
            <div class="row tests"><span>PSI</span><span colspan="2">{ps['value']} ({ps['severity']}) &nbsp; {'⚠ drifted' if ps['drifted'] else '✓ ok'}</span></div>
            """
        else:
            chi = r["chi2"]
            ref_cats = "  ".join([f"{k}: {round(v*100)}%" for k, v in list(r["ref_top"].items())[:3]])
            curr_cats = "  ".join([f"{k}: {round(v*100)}%" for k, v in list(r["curr_top"].items())[:3]])
            detail = f"""
            <div class="row"><span>Top (ref)</span><span style="grid-column:span 2">{ref_cats}</span></div>
            <div class="row"><span>Top (curr)</span><span style="grid-column:span 2">{curr_cats}</span></div>
            <div class="row tests"><span>Chi2</span><span colspan="2">stat={chi['stat']} &nbsp; p={chi['p']} &nbsp; {'⚠ drifted' if chi['drifted'] else '✓ ok'}</span></div>
            """

        cards += f"""
        <div class="card" style="border-top:3px solid {color}">
          <div class="card-head">
            <span class="col-name">{col}</span>
            <span class="badge" style="background:{color}">{label}</span>
          </div>
          <div class="col-type">{r['type']}</div>
          <div class="grid-head"><span>Metric</span><span>Reference</span><span>Current</span></div>
          {detail}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Drift Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f1f5f9; padding: 2rem; color: #1e293b; }}
  h1 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: .25rem; }}
  .meta {{ color: #64748b; font-size: .85rem; margin-bottom: 1.5rem; }}
  .summary {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .box {{ background: white; border-radius: 10px; padding: 1rem 1.5rem; border: 1px solid #e2e8f0; min-width: 120px; }}
  .box .n {{ font-size: 2rem; font-weight: 700; }}
  .box .l {{ font-size: .75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1rem; }}
  .card {{ background: white; border-radius: 10px; padding: 1rem 1.25rem; border: 1px solid #e2e8f0; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .25rem; }}
  .col-name {{ font-weight: 600; font-size: 1rem; }}
  .badge {{ color: white; font-size: .7rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; }}
  .col-type {{ font-size: .75rem; color: #94a3b8; margin-bottom: .75rem; }}
  .grid-head {{ display: grid; grid-template-columns: 1fr 1fr 1fr; font-size: .75rem; color: #94a3b8; font-weight: 500; padding: 4px 0; border-bottom: 1px solid #f1f5f9; margin-bottom: 2px; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; font-size: .85rem; padding: 4px 0; border-bottom: 1px solid #f8fafc; }}
  .row.tests {{ font-family: monospace; font-size: .78rem; color: #475569; grid-template-columns: 80px 1fr; }}
</style>
</head>
<body>
  <h1>Drift Report</h1>
  <div class="meta">Reference: <b>{ref_path}</b> &nbsp;|&nbsp; Current: <b>{curr_path}</b> &nbsp;|&nbsp; {ts}</div>
  <div class="summary">
    <div class="box"><div class="n">{len(results)}</div><div class="l">Columns</div></div>
    <div class="box"><div class="n" style="color:#ef4444">{drifted}</div><div class="l">Drifted</div></div>
    <div class="box"><div class="n" style="color:#22c55e">{stable}</div><div class="l">Stable</div></div>
  </div>
  <div class="cards">{cards}</div>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    required=True)
    parser.add_argument("--curr",   required=True)
    parser.add_argument("--output", default="drift_report.html")
    args = parser.parse_args()

    try:
        ref_df  = pd.read_csv(args.ref)
        curr_df = pd.read_csv(args.curr)
    except FileNotFoundError as e:
        print(f"Error: {e}"); sys.exit(1)

    results = analyze(ref_df, curr_df)
    build_html(results, args.ref, args.curr, args.output)


if __name__ == "__main__":
    main()
