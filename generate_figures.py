"""Generate the publication figures of the CBMP report with cnsplots.

All figures are drawn with the cnsplots plotting API (cns.figure,
cns.lineplot, cns.barplot, cns.scatterplot, cns.savefig) from the CSV files in
``result_csv/``. Unless stated otherwise, the configuration with full symmetry
breaking (rotation + reflection) is used.

Usage:  python generate_figures.py
Output: report/figures/fig{1..7}_*.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cnsplots as cns  # noqa: E402

from classes.graph import load_graphs  # noqa: E402

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
            "gurobi_mip", "cplex_mip", "highs_mip"]
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
    ax.text(18, 0.22, "below diagonal:\nCP-SAT faster", fontsize=7, color="dimgray")
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


def fig6_primal_dual(dfs: dict) -> None:
    graphs = load_graphs(
        root=os.path.join(BASE_DIR, "dataset"),
        skipped=[],
        max_clauses=50_000_000,
        max_edges=None,
        max_density=0.8,
        sort_by_edges=True,
    )
    vertex_counts = {g.name: g.get_vertex_count() for g in graphs}
    all_names = sorted(vertex_counts.keys())

    # Best UB and best LB found across all 8 solvers under full symmetry breaking
    best_ub, best_lb = {}, {}
    for g in all_names:
        ubs, lbs = [], []
        for key in SOLVERS:
            if key not in dfs:
                continue
            sub = full_sym(dfs[key])
            hit = sub[sub["graph_name"] == g]
            if not hit.empty:
                r = hit.iloc[0]
                if pd.notna(r["bandwidth"]):
                    ubs.append(int(r["bandwidth"]))
                if pd.notna(r["lower_bound"]):
                    lbs.append(int(r["lower_bound"]))
        best_ub[g] = min(ubs)
        best_lb[g] = max(lbs)

    rows = []
    for key, label in SOLVERS.items():
        if key not in dfs:
            continue
        sub = full_sym(dfs[key])
        opt_cnt = int((sub["status"] == "OPTIMAL").sum())
        ub_gaps, lb_gaps = [], []
        for _, r in sub.iterrows():
            g = r["graph_name"]
            b_val = float(r["bandwidth"]) if pd.notna(r["bandwidth"]) else float(vertex_counts[g] // 2)
            ub_gaps.append((b_val - best_ub[g]) / best_ub[g] * 100.0)
            lb_val = float(r["lower_bound"])
            lb_gaps.append((best_lb[g] - lb_val) / best_lb[g] * 100.0)
        rows.append({
            "Solver": label,
            "UBGap": float(np.mean(ub_gaps)),
            "LBGap": float(np.mean(lb_gaps)),
            "OPT": opt_cnt,
        })
    data = pd.DataFrame(rows)
    hue_order = list(data["Solver"])

    fig = cns.figure(7.2, 4.1, unit="in")
    ax = fig.add_subplot(111)

    # Dashed Pareto frontier connecting the 3 non-dominated solvers
    frontier_names = ["CPLEX CP", "OR-Tools CP-SAT", "SAT (Order)"]
    f_pts = data.set_index("Solver").loc[frontier_names]
    ax.plot(
        f_pts["UBGap"], f_pts["LBGap"],
        color="#555555", ls="--", lw=1.0, alpha=0.65, zorder=1,
        label="Pareto frontier",
    )

    cns.scatterplot(
        data=data, x="UBGap", y="LBGap",
        hue="Solver", style="Solver",
        hue_order=hue_order, style_order=hue_order,
        palette={s: COLORS[s] for s in hue_order},
        markers={s: MARKERS[s] for s in hue_order},
        s=95, legend=False, zorder=3, ax=ax,
    )

    # Custom non-overlapping offsets (in points) for each solver label
    offsets = {
        "CPLEX CP": (10, 12, "left"),
        "OR-Tools CP-SAT": (-11, -8, "right"),
        "SAT (Order)": (10, -2, "left"),
        "SAT (Pairwise)": (10, -4, "left"),
        "SAT (Incremental)": (-11, 10, "right"),
        "Gurobi MIP": (-8, -18, "center"),
        "CPLEX MIP": (6, -18, "center"),
        "HiGHS MIP": (-12, 16, "center"),
    }
    for _, r in data.iterrows():
        s = r["Solver"]
        dx, dy, ha = offsets.get(s, (8, 4, "left"))
        ax.annotate(
            f"{s}\n({int(r['OPT'])} OPT)",
            xy=(r["UBGap"], r["LBGap"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha, va="center", fontsize=7.5, color="#222222",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.3, 600)
    ax.set_ylim(0, 26.5)
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t}%" for t in ticks]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.annotate(
        "Ideal (0%, 0%)\n(best UB & best LB)",
        xy=(0.38, 0.8), xytext=(1.3, 3.6),
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.9),
        fontsize=7.5, color="dimgray", ha="left", va="center",
    )
    ax.text(5.0, 13.0, "Pareto frontier", fontsize=7.2, color="#555555", rotation=-14)
    ax.set_xlabel(r"Mean relative UB gap to $B_{\mathrm{best}}$ (log scale, TO $= \lfloor n/2\rfloor$, lower is better)")
    ax.set_ylabel(r"Mean relative LB gap to $\mathrm{LB}_{\mathrm{best}}$ (lower is better)")
    ax.grid(True, which="major", ls=":", alpha=0.5)
    save(fig, "fig6_primal_dual.png")


def fig7_sat_build_solve(dfs: dict) -> None:
    sat_keys = ["sat_order", "sat_pairwise", "sat_incremental"]
    labels = [SOLVERS[k] for k in sat_keys]

    opt_build, opt_solve, opt_share, opt_calls = [], [], [], []
    all_build, all_max_build, all_calls = [], [], []
    for k in sat_keys:
        sub = full_sym(dfs[k]).copy()
        sub["calls"] = sub["extra"].apply(lambda x: json.loads(x).get("solve_calls", 0) if pd.notna(x) else 0)
        opt = sub[sub["status"] == "OPTIMAL"]
        opt_build.append(float(opt["build_time"].mean()))
        opt_solve.append(float(opt["solve_time"].mean()))
        opt_share.append(float((opt["build_time"] / opt["total_time"]).mean() * 100.0))
        opt_calls.append(float(opt["calls"].mean()))
        all_build.append(float(sub["build_time"].mean()))
        all_max_build.append(float(sub["build_time"].max()))
        all_calls.append(float(sub["calls"].mean()))

    fig = cns.figure(8.0, 3.6, unit="in")
    x = np.arange(len(labels))
    width = 0.52

    # Panel (a): Stacked build vs solve time on certified OPT instances
    ax1 = fig.add_subplot(121)
    ax1.bar(x, opt_build, width, label="CNF build time", color="#3182bd", edgecolor="black", linewidth=0.5)
    ax1.bar(x, opt_solve, width, bottom=opt_build, label="CaDiCaL solve time", color="#bdd7e7", edgecolor="black", linewidth=0.5)
    for idx in range(len(labels)):
        tot = opt_build[idx] + opt_solve[idx]
        ax1.text(
            x[idx], tot + 1.8,
            f"{tot:.1f} s\n(build {opt_share[idx]:.1f}%)",
            ha="center", va="bottom", fontsize=7.5,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("Mean time per certified OPT instance (s)")
    ax1.set_ylim(0, max(b + s for b, s in zip(opt_build, opt_solve)) * 1.24)
    ax1.set_title("(a) Certified OPT time breakdown", fontsize=9.5, pad=6)
    ax1.grid(True, axis="y", ls=":", alpha=0.5)
    ax1.legend(loc="upper left", frameon=False, fontsize=8)

    # Panel (b): Mean CNF build time across all 59 instances + mean SAT calls
    ax2 = fig.add_subplot(122)
    bar_colors = [COLORS[lbl] for lbl in labels]
    ax2.bar(x, all_build, width, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    for idx in range(len(labels)):
        ax2.text(
            x[idx], all_build[idx] + 0.4,
            f"{all_build[idx]:.2f} s\n(max {all_max_build[idx]:.1f} s)\n{all_calls[idx]:.1f} calls",
            ha="center", va="bottom", fontsize=7.2,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel("Mean CNF build time over all 59 instances (s)")
    ax2.set_ylim(0, max(all_build) * 1.38)
    ax2.set_title("(b) Overall CNF build overhead & SAT calls", fontsize=9.5, pad=6)
    ax2.grid(True, axis="y", ls=":", alpha=0.5)

    fig.tight_layout(w_pad=2.2)
    save(fig, "fig7_sat_build_solve.png")


def main() -> None:
    dfs = load_results()
    print(f"Loaded {len(dfs)} solver result files from {RESULTS_DIR}")
    fig1_cactus(dfs)
    fig2_symmetry(dfs)
    fig3_bandwidth(dfs)
    fig4_scatter(dfs)
    fig5_scaling(dfs)
    fig6_primal_dual(dfs)
    fig7_sat_build_solve(dfs)
    print(f"All figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
