import csv
import json
import math
import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# ============================================================
# 0. BASE DIR AND OUTPUT DIR
# ============================================================

base_dir = Path("/home/users/fdollaku/circuit-breakers")
out_dir = base_dir / "comparison_results"
out_dir.mkdir(parents=True, exist_ok=True)

for old in out_dir.glob("llama_mistral_plot_comparison_*.csv"):
    old.unlink(missing_ok=True)

# ============================================================
# 1. PATHS TO YOUR HTML FILES ON IRIS
# ============================================================

llama_base = base_dir / "IHL_training" / "checking_ihl_compatability"
mistral_base = base_dir / "mistral_IHL_training"

files = {
    "length": {
        "llama":   llama_base   / "llama_length_comparison.html",
        "mistral": mistral_base / "length_comparison.html",
    },
    "similarity": {
        "llama":   llama_base   / "llama_similarity_comparison.html",
        "mistral": mistral_base / "similarity_comparison.html",
    },
    "refusal_rate": {
        "llama":   llama_base   / "llama_refusal_rate.html",
        "mistral": mistral_base / "refusal_rate.html",
    },
}

print("Llama HTML dir:", llama_base)
for p in sorted(llama_base.glob("*.html")):
    print("  Llama:", p.name)

print("\nMistral HTML dir:", mistral_base)
for p in sorted(mistral_base.glob("*.html")):
    print("  Mistral:", p.name)

# ============================================================
# 2. EXTRACT PLOTLY DATA FROM OFFLINE HTML
# ============================================================

plotly_json_pattern = re.compile(
    r"Plotly\.newPlot\([^,]+,\s*(\[\{.*?\}\]),",
    re.DOTALL,
)

def load_traces_from_html(html_path: Path):
    if not html_path.is_file():
        raise FileNotFoundError(f"File not found: {html_path}")
    text = html_path.read_text(encoding="utf-8")
    m = plotly_json_pattern.search(text)
    if not m:
        raise ValueError(f"Could not find Plotly JSON in {html_path}")
    data_json = m.group(1)
    data = json.loads(data_json)
    return data

def summarize_y(values):
    if not values:
        return 0, math.nan, math.nan, math.nan
    n = len(values)
    mean = sum(values) / n
    ymin = min(values)
    ymax = max(values)
    return n, mean, ymin, ymax

# ============================================================
# 3. RAW STATS PER TRACE
# ============================================================

rows = []

for plot_type, models in files.items():
    traces_by_model = {name: load_traces_from_html(path) for name, path in models.items()}
    for model_name, traces in traces_by_model.items():
        for t in traces:
            y = [float(v) for v in t.get("y", [])]
            n_points, y_mean, y_min, y_max = summarize_y(y)
            trace_name = t.get("name", "trace")
            rows.append(
                {
                    "plot_type": plot_type,
                    "model": model_name,
                    "trace_name": trace_name,
                    "n_points": n_points,
                    "y_mean": y_mean,
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )

if not rows:
    raise SystemExit("No data loaded – check paths/filenames.")

print("\n=== Raw stats per trace ===")
for r in rows:
    print(
        f"{r['plot_type']:12s} {r['model']:8s} {r['trace_name']:22s} "
        f"n={r['n_points']:3d} mean={r['y_mean']:.6f} "
        f"min={r['y_min']:.6f} max={r['y_max']:.6f}"
    )

# ============================================================
# 4. BUILD Llama vs Mistral COMPARISON WITH DELTAS
# ============================================================

index = {}
for r in rows:
    key = (r["plot_type"], r["trace_name"])
    index.setdefault(key, {})
    index[key][r["model"]] = r

comparison_rows = []
for (plot_type, trace_name), per_model in index.items():
    if "llama" not in per_model or "mistral" not in per_model:
        continue
    l = per_model["llama"]
    m = per_model["mistral"]
    y_mean_delta = m["y_mean"] - l["y_mean"]
    y_mean_rel_pct = (y_mean_delta / l["y_mean"] * 100.0) if l["y_mean"] != 0 else math.nan
    comparison_rows.append(
        {
            "plot_type": plot_type,
            "trace_name": trace_name,
            "llama_y_mean": l["y_mean"],
            "mistral_y_mean": m["y_mean"],
            "delta_y_mean_abs": y_mean_delta,
            "delta_y_mean_rel_pct": y_mean_rel_pct,
            "llama_y_min": l["y_min"],
            "mistral_y_min": m["y_min"],
            "llama_y_max": l["y_max"],
            "mistral_y_max": m["y_max"],
        }
    )

print("\n=== Llama vs Mistral (mean deltas) ===")
for r in comparison_rows:
    print(
        f"{r['plot_type']:12s} {r['trace_name']:22s} "
        f"L_mean={r['llama_y_mean']:.6f}  "
        f"M_mean={r['mistral_y_mean']:.6f}  "
        f"Δ={r['delta_y_mean_abs']:.6f}  "
        f"Δ%={r['delta_y_mean_rel_pct']:.2f}%"
    )

# ============================================================
# 5. WRITE CSVs
# ============================================================

raw_csv = out_dir / "llama_mistral_plot_comparison_raw.csv"
pivot_csv = out_dir / "llama_mistral_plot_comparison_pivot.csv"

with raw_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "plot_type",
            "model",
            "trace_name",
            "n_points",
            "y_mean",
            "y_min",
            "y_max",
        ],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

with pivot_csv.open("w", newline="") as f:
    if comparison_rows:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        for r in comparison_rows:
            writer.writerow(r)

# ============================================================
# 6. HTML TABLE OF COMPARISON
# ============================================================

table_html = out_dir / "llama_mistral_comparison_table.html"

def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

with table_html.open("w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'><title>Llama vs Mistral comparison</title></head><body>\n")
    f.write("<h1>Llama vs Mistral comparison</h1>\n")
    f.write("<table border='1' cellpadding='4' cellspacing='0'>\n")
    f.write("<tr><th>Plot type</th><th>Trace</th>"
            "<th>Llama mean</th><th>Mistral mean</th>"
            "<th>Δ mean</th><th>Δ mean %</th>"
            "<th>Llama min</th><th>Mistral min</th>"
            "<th>Llama max</th><th>Mistral max</th></tr>\n")
    for r in comparison_rows:
        f.write(
            "<tr>"
            f"<td>{esc(r['plot_type'])}</td>"
            f"<td>{esc(r['trace_name'])}</td>"
            f"<td>{r['llama_y_mean']:.6f}</td>"
            f"<td>{r['mistral_y_mean']:.6f}</td>"
            f"<td>{r['delta_y_mean_abs']:.6f}</td>"
            f"<td>{r['delta_y_mean_rel_pct']:.2f}</td>"
            f"<td>{r['llama_y_min']:.6f}</td>"
            f"<td>{r['mistral_y_min']:.6f}</td>"
            f"<td>{r['llama_y_max']:.6f}</td>"
            f"<td>{r['mistral_y_max']:.6f}</td>"
            "</tr>\n"
        )
    f.write("</table>\n</body></html>\n")

print("Wrote HTML comparison table to", table_html)

# ============================================================
# 7. COMPARISON BAR CHART (Plotly)
# ============================================================

# For each (plot_type, trace), we create a grouped bar: Llama vs Mistral mean
categories = []
llama_means = []
mistral_means = []

for r in comparison_rows:
    label = f"{r['plot_type']} – {r['trace_name']}"
    categories.append(label)
    llama_means.append(r["llama_y_mean"])
    mistral_means.append(r["mistral_y_mean"])

fig = go.Figure()
fig.add_bar(name="Llama", x=categories, y=llama_means)
fig.add_bar(name="Mistral", x=categories, y=mistral_means)
fig.update_layout(
    barmode="group",
    title="Llama vs Mistral – mean values per plot/trace",
    xaxis_title="Plot type / trace",
    yaxis_title="Mean value",
)

plot_html = out_dir / "llama_mistral_comparison_plot.html"
pio.write_html(fig, file=str(plot_html), auto_open=False)  # saves standalone HTML [web:132][web:135][web:137][web:138]

print("Wrote comparison plot to", plot_html)
print("\nAll outputs are in:", out_dir)

