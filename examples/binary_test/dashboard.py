"""Generate an interactive HTML dashboard from run_test.log results.

Usage::

    uv run python examples/binary_test/dashboard.py
    # → opens examples/binary_test/dashboard.html
"""

import math
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOG = Path(__file__).parent / "run_test.log"
OUT = Path(__file__).parent / "dashboard.html"

# --------------------------------------------------------------------------- #
# Parse log
# --------------------------------------------------------------------------- #
rows = []
for line in LOG.read_text().splitlines():
    line = line.strip()
    if not line.startswith("OK"):
        continue
    p = line.split()
    lle_s = p[5].split("=")[1].rstrip("%")
    vle_s = p[6].split("=")[1].rstrip("%")
    rows.append(
        dict(
            sys=p[1],
            water=p[2],
            scen=p[3],
            order=int(p[4].split("=")[1]),
            lle=float(lle_s) if lle_s != "nan" else float("nan"),
            vle=float(vle_s) if vle_s != "nan" else float("nan"),
        )
    )

df = pd.DataFrame(rows)
SYSTEMS = ["mibk_water", "2m2boh_water", "2m3b2oh_water"]
SCEN_COLORS = {"lle": "#E74C3C", "vle": "#3498DB", "vle+lle": "#27AE60"}
ORDER_SYMBOLS = {0: "circle", 1: "triangle-up"}
ORDER_DASH = {0: "solid", 1: "dot"}
CAP = 500.0  # cap ARD for color scale

# --------------------------------------------------------------------------- #
# Figure 1 — Scatter: ARD_LLE vs ARD_VLE  (one subplot per system)
# --------------------------------------------------------------------------- #
fig1 = make_subplots(
    rows=1, cols=3,
    subplot_titles=[s.replace("_", " ") for s in SYSTEMS],
    shared_xaxes=False, shared_yaxes=False,
    horizontal_spacing=0.08,
)

seen_legend = set()

for col, sys in enumerate(SYSTEMS, start=1):
    sub = df[df.sys == sys]
    for scen in ["lle", "vle", "vle+lle"]:
        for order in [0, 1]:
            chunk = sub[(sub.scen == scen) & (sub.order == order)]
            key = (scen, order)
            show = key not in seen_legend
            if show:
                seen_legend.add(key)
            fig1.add_trace(
                go.Scatter(
                    x=chunk["lle"],
                    y=chunk["vle"],
                    mode="markers",
                    marker=dict(
                        color=SCEN_COLORS[scen],
                        symbol=ORDER_SYMBOLS[order],
                        size=9,
                        opacity=0.85,
                        line=dict(width=1, color="white"),
                    ),
                    name=f"{scen} ord={order}",
                    legendgroup=f"{scen}_ord{order}",
                    showlegend=show,
                    text=chunk["water"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"scenario: {scen}  order: {order}<br>"
                        "ARD LLE: %{x:.1f}%<br>"
                        "ARD VLE: %{y:.1f}%<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
    # diagonal reference line (equal LLE and VLE)
    ax_max = min(CAP, 300)
    fig1.add_trace(
        go.Scatter(
            x=[0.5, ax_max], y=[0.5, ax_max],
            mode="lines",
            line=dict(color="lightgray", dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=col,
    )
    fig1.update_xaxes(
        title_text="ARD LLE (%)", type="log",
        range=[np.log10(0.5), np.log10(CAP)],
        row=1, col=col,
    )
    fig1.update_yaxes(
        title_text="ARD VLE (%)", type="log",
        range=[np.log10(0.5), np.log10(CAP)],
        row=1, col=col,
    )

fig1.update_layout(
    title="LLE vs VLE ARD — all water models, scenarios & orders<br>"
          "<sup>Closer to origin = better overall. Diagonal = equal LLE/VLE quality.</sup>",
    height=520,
    legend=dict(
        title="Scenario / Order",
        orientation="v",
        x=1.01, y=1,
    ),
    template="plotly_white",
    font=dict(size=12),
)

# --------------------------------------------------------------------------- #
# Figure 2 — Heatmaps: water model × scenario  (LLE and VLE side-by-side)
# --------------------------------------------------------------------------- #
COLS_HEAT = [
    ("lle", 0), ("lle", 1),
    ("vle", 0), ("vle", 1),
    ("vle+lle", 0), ("vle+lle", 1),
]
COL_LABELS = [f"{s}<br>ord={o}" for s, o in COLS_HEAT]

# Sort water models by mean combined ARD across all scenarios (ascending = best first)
water_order = {}
for sys in SYSTEMS:
    sub = df[(df.sys == sys) & ~df.lle.isna() & ~df.vle.isna()]
    mean_comb = (
        sub.groupby("water")
        .apply(lambda g: (g["lle"].clip(upper=CAP) + g["vle"].clip(upper=CAP)).mean())
        .sort_values()
    )
    water_order[sys] = mean_comb.index.tolist()

fig2 = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        f"{s.replace('_',' ')}  —  ARD LLE (%)" if i % 2 == 0
        else f"{s.replace('_',' ')}  —  ARD VLE (%)"
        for s in SYSTEMS for i in range(2)
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.12,
)

for row_idx, sys in enumerate(SYSTEMS, start=1):
    sub = df[df.sys == sys]
    waters = water_order[sys]

    for side, metric in enumerate(["lle", "vle"], start=1):
        z = []
        text_z = []
        for w in waters:
            row_vals = []
            row_text = []
            for scen, order in COLS_HEAT:
                match = sub[(sub.water == w) & (sub.scen == scen) & (sub.order == order)]
                val = float(match[metric].values[0]) if len(match) else float("nan")
                row_vals.append(min(val, CAP) if not math.isnan(val) else float("nan"))
                row_text.append(f"{val:.1f}%" if not math.isnan(val) else "n/a")
            z.append(row_vals)
            text_z.append(row_text)

        z_arr = np.array(z, dtype=float)
        # log-transform for color scale (add 1 to handle near-zero)
        z_log = np.where(np.isnan(z_arr), np.nan, np.log10(np.maximum(z_arr, 0.1)))

        fig2.add_trace(
            go.Heatmap(
                z=z_log,
                x=COL_LABELS,
                y=waters,
                text=text_z,
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorscale="RdYlGn_r",
                zmin=np.log10(0.5),
                zmax=np.log10(CAP),
                colorbar=dict(
                    title="ARD %<br>(log)",
                    tickvals=[np.log10(v) for v in [1, 5, 10, 50, 100, 500]],
                    ticktext=["1", "5", "10", "50", "100", "500"],
                    len=0.25,
                    y=1.0 - (row_idx - 1) * 0.37,
                    x=1.0 if side == 2 else None,
                    xanchor="left" if side == 2 else "right",
                    thickness=12,
                ) if side == 2 else dict(showticklabels=False, thickness=0),
                showscale=(side == 2),
                hovertemplate=(
                    f"<b>%{{y}}</b><br>"
                    f"{'LLE' if metric=='lle' else 'VLE'} ARD: %{{text}}<br>"
                    "Scenario: %{x}<extra></extra>"
                ),
                xgap=2,
                ygap=1,
            ),
            row=row_idx,
            col=side,
        )

fig2.update_layout(
    title="ARD heatmap by water model and scenario<br>"
          "<sup>Green = low ARD (good), Red = high ARD (bad). Water models sorted best→worst by mean combined score.</sup>",
    height=180 + 270 * len(SYSTEMS),
    template="plotly_white",
    font=dict(size=10),
)
for row_idx in range(1, len(SYSTEMS) + 1):
    for col_idx in range(1, 3):
        fig2.update_xaxes(tickangle=-30, row=row_idx, col=col_idx)
        fig2.update_yaxes(tickfont=dict(size=9), row=row_idx, col=col_idx)

# --------------------------------------------------------------------------- #
# Figure 3 — Best-per-system comparison bar chart (top 8 water models)
# --------------------------------------------------------------------------- #
fig3 = make_subplots(
    rows=1, cols=3,
    subplot_titles=[s.replace("_", " ") for s in SYSTEMS],
    shared_yaxes=False,
)

for col, sys in enumerate(SYSTEMS, start=1):
    sub_combined = df[(df.sys == sys) & (df.scen == "vle+lle") & (df.order == 0)].copy()
    sub_combined["sum"] = sub_combined["lle"].clip(upper=CAP) + sub_combined["vle"].clip(upper=CAP)
    top8 = sub_combined.nsmallest(8, "sum")["water"].tolist()

    sub = df[(df.sys == sys) & (df.water.isin(top8)) & (df.order == 0)]
    sub = sub.set_index("water").loc[top8].reset_index()

    for scen in ["lle", "vle", "vle+lle"]:
        chunk = sub[sub.scen == scen]
        show = col == 1
        for metric, opacity, pattern in [("lle", 1.0, ""), ("vle", 0.45, "/")]:
            fig3.add_trace(
                go.Bar(
                    name=f"ARD {'LLE' if metric=='lle' else 'VLE'} ({scen})",
                    x=chunk["water"],
                    y=chunk[metric].clip(upper=CAP),
                    marker=dict(
                        color=SCEN_COLORS[scen],
                        opacity=opacity,
                        pattern_shape=pattern,
                    ),
                    legendgroup=f"{metric}_{scen}",
                    showlegend=show,
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>{metric.upper()} ARD ({scen}): %{{y:.1f}}%<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
    fig3.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=1, col=col)
    fig3.update_yaxes(title_text="ARD (%)", range=[0, CAP], row=1, col=col)

fig3.update_layout(
    title="Top-8 water models per system — scenario comparison (order=0, capped at 500%)<br>"
          "<sup>Solid bar = ARD LLE, hatched = ARD VLE. Sorted by best vle+lle combined score.</sup>",
    height=480,
    barmode="group",
    legend=dict(orientation="h", y=-0.28, x=0),
    template="plotly_white",
    font=dict(size=11),
)

# --------------------------------------------------------------------------- #
# Write HTML
# --------------------------------------------------------------------------- #
html_parts = [
    "<html><head><meta charset='utf-8'>",
    "<title>Binary k_ij Fitting Dashboard</title>",
    "<style>body{font-family:sans-serif;margin:20px;background:#f9f9f9}"
    "h1{color:#2c3e50}h2{color:#7f8c8d;margin-top:40px}.fig{background:white;"
    "padding:10px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1);margin-bottom:30px}"
    "</style></head><body>",
    "<h1>Binary k_ij Fitting — Water Model Benchmark</h1>",
    "<p>3 systems &times; 27 water models &times; 3 scenarios (LLE-only, VLE-only, VLE+LLE) "
    "&times; order 0 &amp; 1. ARD% is evaluated at the fitted polynomial for <em>both</em> "
    "data types (cross-evaluation for single-type scenarios).</p>",
    "<h2>1 — ARD LLE vs ARD VLE  (log–log scatter)</h2><div class='fig'>",
    fig1.to_html(full_html=False, include_plotlyjs="cdn"),
    "</div><h2>2 — Heatmap: all water models &times; scenarios</h2><div class='fig'>",
    fig2.to_html(full_html=False, include_plotlyjs=False),
    "</div><h2>3 — Top-8 water models per system</h2><div class='fig'>",
    fig3.to_html(full_html=False, include_plotlyjs=False),
    "</div></body></html>",
]

OUT.write_text("\n".join(html_parts), encoding="utf-8")
print(f"Saved → {OUT}")
webbrowser.open(OUT.as_uri())
