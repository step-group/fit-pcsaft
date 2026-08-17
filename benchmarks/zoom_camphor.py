import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = json.load(open("benchmarks/camphor-transfer/fronts.json"))
COL = {"sbx": "#1f77b4", "de-cr1.0": "#ff7f0e"}
rungs = [2700, 10800, 28800]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
for ax, ev in zip(axes, rungs):
    for op in ("sbx", "de-cr1.0"):
        first = True
        for r in rows:
            if r["op"] != op or r["evals"] != ev:
                continue
            F = np.asarray(r["F"])
            F = F[(F[:, 0] <= 10) & (F[:, 1] <= 10)]
            if len(F) == 0:
                continue
            F = F[np.argsort(F[:, 0])]
            ax.plot(F[:, 0], F[:, 1], "o-", ms=4, lw=0.9, alpha=0.8,
                    color=COL[op], label=op if first else None)
            first = False
    ax.set_title(f"{ev:,} evals/cell")
    ax.set_xlabel("AARD_psat [%]")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("AARD_rho [%]")
axes[0].legend(loc="upper right")
fig.suptitle("camphor fronts, zoomed to AARD <= 10 % (3 seeds per operator)")
fig.tight_layout()
fig.savefig("benchmarks/camphor-transfer/zoom_low_aard.png", dpi=140, bbox_inches="tight")
print("wrote benchmarks/camphor-transfer/zoom_low_aard.png")
