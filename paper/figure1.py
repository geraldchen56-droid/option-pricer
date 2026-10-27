"""Figure 1: standard error and RMSE against number of paths (log-log).

Run convergence.py first; this reads results/convergence_summary.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = Path(__file__).parent
s = json.loads((HERE / "results" / "convergence_summary.json").read_text())
n = np.array([x["n"] for x in s["rows"]])
se = np.array([x["mean_se"] for x in s["rows"]])
rmse = np.array([x["rmse"] for x in s["rows"]])

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.0, 3.9))

grid = np.logspace(3, 6, 50)
ax.plot(grid, se[0] * np.sqrt(n[0]) / np.sqrt(grid), ls="--", color="0.55", lw=1.2,
        label=r"Theory: slope $-1/2$ (lies under the data)")
ax.plot(n, se, "o-", color="#1f4e8c", lw=1.8, ms=5,
        label=f"Mean standard error (fitted slope $-${abs(s['slope_se']):.3f})")
ax.plot(n, rmse, "s", color="0.35", ms=4.5, mfc="white", mew=1.1, label="RMSE vs Black-Scholes")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(7e2, 1.5e6)
ax.set_ylim(0.01, 0.8)
ax.yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Number of simulated paths, $N$")
ax.set_ylabel("Pricing error (price units)")
ax.grid(True, which="major", color="0.9", lw=0.6)
ax.legend(frameon=False, loc="lower left", fontsize=10)
ax.annotate(f"{se[0]:.3f}", (n[0], se[0]), xytext=(8, 4), textcoords="offset points",
            fontsize=9.5, color="#1f4e8c")
ax.annotate(f"{se[-1]:.4f}", (n[-1], se[-1]), xytext=(-46, -4), textcoords="offset points",
            fontsize=9.5, color="#1f4e8c")

fig.tight_layout()
fig.savefig(HERE / "results" / "figure1.pdf")
fig.savefig(HERE / "results" / "figure1.png", dpi=200)
print("saved results/figure1.pdf and results/figure1.png")
