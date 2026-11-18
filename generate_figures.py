"""Generate the publication figures of the CBMP report with cnsplots.

All figures are drawn with the cnsplots plotting API (cns.figure,
cns.lineplot, cns.barplot, cns.scatterplot, cns.savefig) from the CSV files in
``result_csv/``. Unless stated otherwise, the configuration with full symmetry
breaking (rotation + reflection) is used.

Usage:  python generate_figures.py
Output: report/figures/fig{1..5}_*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cnsplots as cns  # noqa: E402

cns.setup_matplotlib()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "result_csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_LIMIT = 300.0
DPI = 300

SOLVERS = {
    "sat_order": "SAT (Order)",
    "sat_pairwise": "SAT (Pairwise)",
    "sat_incremental": "SAT (Incremental)",
    "cpsat": "OR-Tools CP-SAT",
    "cplex_cp": "CPLEX CP",
    "gurobi_mip": "Gurobi MIP",
    "cplex_mip": "CPLEX MIP",
    "highs_mip": "HiGHS MIP",
}
COLORS = {
    "SAT (Order)": "#1f77b4",
    "SAT (Pairwise)": "#6baed6",
    "SAT (Incremental)": "#2ca02c",
    "OR-Tools CP-SAT": "#ff7f0e",
    "CPLEX CP": "#9467bd",
    "Gurobi MIP": "#d62728",
    "CPLEX MIP": "#8c564b",
    "HiGHS MIP": "#7f7f7f",
}
MARKERS = {
    "SAT (Order)": "o",
    "SAT (Pairwise)": "s",
    "SAT (Incremental)": "^",
    "OR-Tools CP-SAT": "D",
    "CPLEX CP": "v",
    "Gurobi MIP": ">",
    "CPLEX MIP": "<",
    "HiGHS MIP": "X",
}


def load_results() -> dict:
    dfs = {}
    for key in SOLVERS:
        path = os.path.join(RESULTS_DIR, f"{key}.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}, skipped")
            continue
        df = pd.read_csv(path)
        for col in ("symmetry_breaking", "reflection_breaking"):
            df[col] = df[col].astype(str).str.lower().eq("true")
        dfs[key] = df
    return dfs


def full_sym(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with rotation + reflection symmetry breaking."""
    return df[df["symmetry_breaking"] & df["reflection_breaking"]]


def save(fig, name: str) -> None:
    cns.savefig(os.path.join(OUTPUT_DIR, name), fig=fig, dpi=DPI)
    plt.close(fig)
    print(f"  saved {name}")


def fig1_cactus(dfs: dict) -> None:
    rows = []
    for key, label in SOLVERS.items():
        if key not in dfs:
            continue
        sub = full_sym(dfs[key])
        times = np.sort(sub.loc[sub["status"] == "OPTIMAL", "total_time"].to_numpy())
        for rank, t in enumerate(times, 1):
            rows.append({"Solver": f"{label} ({len(times)})", "base": label,
                         "Instances": rank, "Time": t})
    data = pd.DataFrame(rows)
    hue_order = list(dict.fromkeys(data["Solver"]))
    base_of = dict(zip(data["Solver"], data["base"]))

    fig = cns.figure(7.0, 4.2, unit="in")
    ax = fig.add_subplot(111)
    cns.lineplot(data=data, x="Instances", y="Time", hue="Solver", style="Solver",
                 hue_order=hue_order, style_order=hue_order,
                 palette={s: COLORS[base_of[s]] for s in hue_order},
                 markers={s: MARKERS[base_of[s]] for s in hue_order},
                 dashes=False, estimator=None, errorbar=None,
                 markersize=5, linewidth=1.5, ax=ax)
    ax.axhline(TIME_LIMIT, color="gray", ls="--", lw=1)
    ax.text(0.5, TIME_LIMIT * 1.12, "time limit (300 s)", color="gray", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_xlabel("Number of instances certified optimal")
    ax.set_ylabel("Total solving time (s)")
    ax.grid(True, which="major", ls=":", alpha=0.5)
    cns.take_legend_out(title="Solver (#OPT)", ax=ax)
    save(fig, "fig1_cactus_runtime.png")


def fig2_symmetry(dfs: dict) -> None:
    keys = ["sat_order", "sat_pairwise", "sat_incremental", "cpsat", "cplex_cp",
            "gurobi_mip", "cplex_mip"]
    setting_name = {(False, False): "None", (True, False): "Rotation",
                    (True, True): "Rotation + Reflection"}
    rows = []
    for key in keys:
        if key not in dfs:
            continue
        for (sb, rb), group in dfs[key].groupby(["symmetry_breaking", "reflection_breaking"]):
            if (sb, rb) in setting_name:
                rows.append({"Solver": SOLVERS[key], "Setting": setting_name[(sb, rb)],
                             "Optimal": int((group["status"] == "OPTIMAL").sum())})
    data = pd.DataFrame(rows)
    order = [SOLVERS[k] for k in keys if k in dfs]
    hue_order = ["None", "Rotation", "Rotation + Reflection"]

    fig = cns.figure(8.0, 3.8, unit="in")
    ax = fig.add_subplot(111)
    cns.barplot(data=data, x="Solver", y="Optimal", hue="Setting", order=order,
                hue_order=hue_order, add_tip=False,
                palette=dict(zip(hue_order, ["#c6dbef", "#4292c6", "#08306b"])),
                edgecolor="black", linewidth=0.5, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=7, padding=1)
    ax.set_xlabel("")
    ax.set_ylabel("Certified optimal instances")
    ax.set_ylim(0, data["Optimal"].max() + 3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    cns.take_legend_out(title="Symmetry breaking", ax=ax)
    save(fig, "fig2_symmetry_impact.png")


SAMPLE_GRAPHS = [
    "bcspwr01", "bcspwr02", "bcsstk01", "curtis54", "will57", "dwt_59",
    "can_62", "dwt_72", "can_73", "ash85", "dwt_87", "can_96", "nos4",
    "bcsstk03", "can_144", "dwt_162",
]


def fig3_bandwidth(dfs: dict) -> None:
    keys = ["sat_order", "cpsat", "cplex_cp", "gurobi_mip"]
    rows = []
    for g in SAMPLE_GRAPHS:
        for key in keys:
            sub = full_sym(dfs[key])
            sub = sub[sub["graph_name"] == g]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({"Graph": g, "Solver": SOLVERS[key],
                         "Bandwidth": r["bandwidth"],
                         "Optimal": r["status"] == "OPTIMAL"})
    data = pd.DataFrame(rows)
    hue_order = [SOLVERS[k] for k in keys]

    fig = cns.figure(9.0, 3.8, unit="in")
    ax = fig.add_subplot(111)
    cns.barplot(data=data, x="Graph", y="Bandwidth", hue="Solver", order=SAMPLE_GRAPHS,
                hue_order=hue_order, palette={s: COLORS[s] for s in hue_order},
                edgecolor="black", linewidth=0.4, ax=ax)
    # one container per hue level, bars follow the x order
    for solver, container in zip(hue_order, ax.containers):
        for g, rect in zip(SAMPLE_GRAPHS, container):
            hit = data[(data["Graph"] == g) & (data["Solver"] == solver)]
            if not hit.empty and bool(hit["Optimal"].iloc[0]):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), "*",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_yscale("log")
    ticks = [2, 3, 5, 10, 20, 50]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(2, 60)
    ax.set_xlabel("")
    ax.set_ylabel("Cyclic bandwidth (log scale, lower is better)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(True, axis="y", which="major", ls=":", alpha=0.5)
    cns.take_legend_out(title="Solver (* = optimal)", ax=ax)
    save(fig, "fig3_bandwidth_comparison.png")


def fig4_scatter(dfs: dict) -> None:
    sat = full_sym(dfs["sat_order"]).set_index("graph_name")
    cps = full_sym(dfs["cpsat"]).set_index("graph_name")
    rows = []
    for g in sorted(set(sat.index) & set(cps.index)):
        s_opt = sat.loc[g, "status"] == "OPTIMAL"
        c_opt = cps.loc[g, "status"] == "OPTIMAL"
        if s_opt and c_opt:
            cat = "Both optimal"
        elif s_opt:
            cat = "SAT (Order) optimal only"
        elif c_opt:
            cat = "CP-SAT optimal only"
        else:
            cat = "Neither optimal"
        rows.append({"Graph": g, "SAT": min(sat.loc[g, "total_time"], TIME_LIMIT),
                     "CPSAT": min(cps.loc[g, "total_time"], TIME_LIMIT), "Category": cat})
    data = pd.DataFrame(rows)
    hue_order = ["Both optimal", "SAT (Order) optimal only", "CP-SAT optimal only",
                 "Neither optimal"]
    hue_order = [h for h in hue_order if h in set(data["Category"])]
    palette = {"Both optimal": "#1f77b4", "SAT (Order) optimal only": "#2ca02c",
               "CP-SAT optimal only": "#ff7f0e", "Neither optimal": "#9e9e9e"}

    fig = cns.figure(5.2, 4.6, unit="in")
    ax = fig.add_subplot(111)
    lims = [0.05, 450]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="y = x")
    cns.scatterplot(data=data, x="SAT", y="CPSAT", hue="Category", hue_order=hue_order,
                    style="Category", style_order=hue_order,
                    markers=dict(zip(["Both optimal", "SAT (Order) optimal only",
                                      "CP-SAT optimal only", "Neither optimal"],
                                     ["o", "^", "s", "X"])),
                    palette=palette, s=40, alpha=0.85, ax=ax)
    ax.axvline(TIME_LIMIT, color="gray", ls=":", lw=0.8)
    ax.axhline(TIME_LIMIT, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("SAT (Order) total time (s)")
    ax.set_ylabel("OR-Tools CP-SAT total time (s)")
    ax.text(0.07, 40, "above diagonal:\nSAT (Order) faster", fontsize=7, color="dimgray")
    ax.text(0.07, 0.5, "below diagonal:\nCP-SAT faster", fontsize=7, color="dimgray")
    ax.grid(True, which="major", ls=":", alpha=0.4)
    cns.take_legend_out(title="Outcome", ax=ax)
    save(fig, "fig4_scatter_sat_vs_cpsat.png")


def fig5_scaling(dfs: dict) -> None:
    keys = ["sat_order", "sat_pairwise", "sat_incremental", "cpsat", "cplex_cp", "gurobi_mip"]
    rows = []
    for key in keys:
        sub = full_sym(dfs[key])
        opt = sub[sub["status"] == "OPTIMAL"]
        for _, r in opt.iterrows():
            rows.append({"Solver": SOLVERS[key], "V": r["vertex_count"], "Time": r["total_time"]})
    data = pd.DataFrame(rows)
    hue_order = [SOLVERS[k] for k in keys]

    fig = cns.figure(7.0, 4.0, unit="in")
    ax = fig.add_subplot(111)
    cns.scatterplot(data=data, x="V", y="Time", hue="Solver", hue_order=hue_order,
                    style="Solver", style_order=hue_order,
                    markers={s: MARKERS[s] for s in hue_order},
                    palette={s: COLORS[s] for s in hue_order}, s=35, alpha=0.8, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Number of vertices |V|")
    ax.set_ylabel("Time to certify optimality (s)")
    ax.grid(True, which="major", ls=":", alpha=0.5)
    cns.take_legend_out(title="Solver", ax=ax)
    save(fig, "fig5_scaling.png")


def main() -> None:
    dfs = load_results()
    print(f"Loaded {len(dfs)} solver result files from {RESULTS_DIR}")
    fig1_cactus(dfs)
    fig2_symmetry(dfs)
    fig3_bandwidth(dfs)
    fig4_scatter(dfs)
    fig5_scaling(dfs)
    print(f"All figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
