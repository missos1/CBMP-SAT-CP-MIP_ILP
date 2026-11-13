"""Central command-line runner for all CBMP benchmark suites."""

import argparse
import os
import subprocess
import sys
from functools import partial

from benchmark.runner import run_benchmark_suite


def main():
    parser = argparse.ArgumentParser(description="CBMP Solver Benchmark Suite")
    parser.add_argument(
        "--solver",
        choices=[
            "cplex_both",
            "cplex_cp",
            "cplex_mip",
            "highs",
            "ortools",
            "gurobi",
            "sat_all",
            "sat_pairwise",
            "sat_incremental",
            "sat_order",
            "all",
        ],
        default="cplex_both",
        help="Which solver benchmark to execute (default: cplex_both)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Timeout in seconds per instance (default: from config.py)",
    )
    args = parser.parse_args()

    s = args.solver
    tl = args.time_limit

    if s == "all":
        for sub_solver in ["cplex_both", "highs", "ortools", "gurobi", "sat_all"]:
            cmd = [sys.executable, os.path.abspath(__file__), "--solver", sub_solver]
            if tl is not None:
                cmd.extend(["--time-limit", str(tl)])
            subprocess.run(cmd, check=False)
        return

    if s in ("cplex_cp", "cplex_both"):
        from benchmark.benchmark_cplex import benchmap_cplex_cp
        run_benchmark_suite(benchmap_cplex_cp, solver_key="cplex_cp", time_limit=tl)

    if s in ("cplex_mip", "cplex_both"):
        from benchmark.benchmark_cplex import benchmap_cplex_mip
        run_benchmark_suite(benchmap_cplex_mip, solver_key="cplex_mip", time_limit=tl)

    if s == "highs":
        from benchmark.benchmark_highs import benchmap_highs
        run_benchmark_suite(benchmap_highs, solver_key="highs_mip", time_limit=tl)

    if s == "ortools":
        from benchmark.benchmark_ortools import benchmap_cpsat
        run_benchmark_suite(benchmap_cpsat, solver_key="cpsat", time_limit=tl)

    if s == "gurobi":
        from benchmark.benchmark_gurobi import benchmap_gurobi_mip
        run_benchmark_suite(benchmap_gurobi_mip, solver_key="gurobi_mip", time_limit=tl)

    if s in ("sat_pairwise", "sat_all"):
        from benchmark.benchmark_sat import benchmap_sat
        fn = partial(benchmap_sat, encoding="pairwise", incremental=False)
        run_benchmark_suite(fn, solver_key="sat_pairwise", encoding="pairwise", time_limit=tl)

    if s in ("sat_incremental", "sat_all"):
        from benchmark.benchmark_sat import benchmap_sat
        fn = partial(benchmap_sat, encoding="pairwise", incremental=True)
        run_benchmark_suite(
            fn, solver_key="sat_incremental", encoding="pairwise_incremental", time_limit=tl
        )

    if s in ("sat_order", "sat_all"):
        from benchmark.benchmark_sat import benchmap_sat
        fn = partial(benchmap_sat, encoding="order", incremental=False)
        run_benchmark_suite(fn, solver_key="sat_order", encoding="order", time_limit=tl)


if __name__ == "__main__":
    main()
