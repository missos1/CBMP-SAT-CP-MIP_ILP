"""CPLEX CP and MIP/ILP baselines for CBMP."""

import math
import time

from docplex.cp.model import CpoModel
from docplex.mp.model import Model

from benchmark.common import (
    ERROR,
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    TIMEOUT,
    UNKNOWN,
    finalize_result,
    theoretical_lower_bound,
)
from classes.graph import Graph


def _cp_status(solution, time_limit: float, total_elapsed: float = 0.0) -> str:
    if solution is None:
        return TIMEOUT if total_elapsed >= max(0.0, time_limit - 1.0) or time_limit <= 1.0 else UNKNOWN
    raw = str(solution.get_solve_status()).lower()
    if "optimal" in raw:
        return OPTIMAL
    if "infeasible" in raw:
        return INFEASIBLE
    if "feasible" in raw:
        return FEASIBLE
    stop_cause = ""
    try:
        stop_cause = str(solution.get_stop_cause()).lower()
    except Exception:
        pass
    if "limit" in stop_cause or "unknown" in raw:
        if "limit" in stop_cause or total_elapsed >= max(0.0, time_limit - 1.0) or time_limit <= 1.0:
            return TIMEOUT
    return UNKNOWN


def _mip_status(solution, details, time_limit: float = 0.0, total_elapsed: float = 0.0) -> str:
    if solution is not None:
        raw = str(solution.solve_status.name).lower()
        if "optimal" in raw:
            return OPTIMAL
        if "infeasible" in raw:
            return INFEASIBLE
        if "feasible" in raw:
            return FEASIBLE
        return FEASIBLE
    raw = str(details.status).lower() if details is not None else ""
    if "infeasible" in raw:
        return INFEASIBLE
    if "time" in raw or "limit" in raw or total_elapsed >= max(0.0, time_limit - 1.0):
        return TIMEOUT
    return UNKNOWN


def benchmap_cplex_cp(graph: Graph, break_symmetry: bool = False,
                      break_reflection: bool = False,
                      time_limit: float = 60.0,
                      workers: int = 8) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="cplex_cp", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    theo_lb = theoretical_lower_bound(graph)
    start = time.perf_counter()

    try:
        mdl = CpoModel(name="cbmp_cplex_cp")
        x = {v: mdl.integer_var(1, n, name=f"x_{i}")
             for i, v in enumerate(vertices)}
        mdl.add(mdl.all_diff(list(x.values())))

        if break_symmetry:
            mdl.add(x[vertices[0]] == 1)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = neighbours[0] if neighbours else vertices[1]
                mdl.add(x[target] <= n // 2 + 1)

        bound = mdl.integer_var(theo_lb, n // 2, name="B")
        for u, v in edges:
            diff = mdl.abs(x[u] - x[v])
            mdl.add((diff <= bound) | (diff >= n - bound))
        mdl.add(mdl.minimize(bound))

        build_time = time.perf_counter() - start
        remaining = max(0.0, float(time_limit) - build_time)
        solve_kwargs = {
            "TimeLimit": remaining,
            # "FailureDirectedSearch": "Off",
            "LogVerbosity": "Quiet",
        }
        if workers is not None and int(workers) > 0:
            solve_kwargs["Workers"] = int(workers)

        solve_start = time.perf_counter()
        fallback_note = ""
        solution = None
        if remaining > 0:
            try:
                solution = mdl.solve(**solve_kwargs)
            except Exception as solve_exc:
                # Resilient fallback if cpoptimizer segfaults (rc=-11 / 0xC0000005)
                elapsed_so_far = time.perf_counter() - start
                rem_fallback = max(1.0, float(time_limit) - elapsed_so_far)
                fallback_workers = 1 if (workers is not None and int(workers) <= 1) else min(8, int(workers) if workers else 8)
                fallback_kwargs = {
                    "TimeLimit": rem_fallback,
                    "FailureDirectedSearch": "Off",
                    "SearchType": "Restart",
                    "Workers": fallback_workers,
                    "LogVerbosity": "Quiet",
                }
                solution = mdl.solve(**fallback_kwargs)
                fallback_note = f"recovered from {type(solve_exc).__name__} (fallback workers={fallback_workers})"

        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start

        status = _cp_status(solution, time_limit, total_time) if remaining > 0 else TIMEOUT
        labels = None
        if solution is not None and solution.is_solution():
            labels = {
                key: int(round(solution.get_value(x[key])))
                for key in vertices
            }

        lower = theo_lb
        if solution is not None:
            try:
                obj_bounds = solution.get_objective_bounds()
                if obj_bounds and obj_bounds[0] is not None and math.isfinite(float(obj_bounds[0])):
                    lower = max(lower, int(math.ceil(float(obj_bounds[0]) - 1e-6)))
            except Exception:
                pass
        if lower > n // 2:
            lower = n // 2

        msg = fallback_note if fallback_note else ("" if solution is not None else "no solution returned")
        return finalize_result(
            solver_name="cplex_cp", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=status, labels=labels,
            lower_bound=lower,
            build_time=build_time, solve_time=solve_time, total_time=total_time,
            message=msg,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return finalize_result(
            solver_name="cplex_cp", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=elapsed, total_time=elapsed,
            message=f"{type(exc).__name__}: {exc}",
        )


def benchmap_cplex_mip(graph: Graph, break_symmetry: bool = False,
                       break_reflection: bool = False,
                       time_limit: float = 60.0,
                       mip_gap: float = 0.0,
                       workers: int = 8) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="cplex_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    index = {v: i for i, v in enumerate(vertices)}
    theo_lb = theoretical_lower_bound(graph)
    start = time.perf_counter()
    mdl = None

    try:
        mdl = Model(name="cbmp_cplex_mip")
        mdl.parameters.mip.tolerances.mipgap = float(mip_gap)
        mdl.parameters.threads = int(workers)

        x = {(i, l): mdl.binary_var(name=f"x_{i}_{l}")
             for i in range(n) for l in range(n)}
        for i in range(n):
            mdl.add_constraint(mdl.sum(x[i, l] for l in range(n)) == 1)
        for l in range(n):
            mdl.add_constraint(mdl.sum(x[i, l] for i in range(n)) == 1)

        if break_symmetry:
            mdl.add_constraint(x[0, 0] == 1)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = index[neighbours[0]] if neighbours else 1
                mdl.add_constraint(
                    mdl.sum(x[target, l] for l in range(n // 2 + 1, n)) == 0
                )

        bound = mdl.integer_var(lb=theo_lb, ub=n // 2, name="B")
        position = mdl.integer_var_list(n, 0, n - 1, name="p")
        for i in range(n):
            mdl.add_constraint(
                position[i] == mdl.sum(l * x[i, l] for l in range(n))
            )

        for edge_index, (u, v) in enumerate(edges):
            iu, iv = index[u], index[v]
            offset = mdl.integer_var(lb=0, ub=n - 1, name=f"d_{edge_index}")
            wrap = mdl.binary_var(name=f"w_{edge_index}")
            side = mdl.binary_var(name=f"s_{edge_index}")
            mdl.add_constraint(position[iv] - position[iu] == offset - n * wrap)
            mdl.add_constraint(offset <= bound + n * (1 - side))
            mdl.add_constraint(n - offset <= bound + n * side)

        mdl.minimize(bound)
        build_time = time.perf_counter() - start
        remaining = max(0.0, float(time_limit) - build_time)
        mdl.parameters.timelimit = remaining

        solve_start = time.perf_counter()
        solution = mdl.solve(log_output=False) if remaining > 0 else None
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start
        details = mdl.solve_details if remaining > 0 else None

        status = _mip_status(solution, details, time_limit, total_time) if remaining > 0 else TIMEOUT
        labels = None
        bandwidth = None
        if solution is not None:
            labels = {}
            for i, key in enumerate(vertices):
                best_l = max(range(n), key=lambda l: solution.get_value(x[i, l]))
                if solution.get_value(x[i, best_l]) > 0.5:
                    labels[key] = best_l + 1
            bandwidth = int(round(solution.get_objective_value()))

        lower = theo_lb
        if details is not None and details.best_bound is not None:
            best_bound = float(details.best_bound)
            if math.isfinite(best_bound) and abs(best_bound) < 1e20:
                lower = max(lower, int(math.ceil(best_bound - 1e-6)))
        if lower > n // 2:
            lower = n // 2

        return finalize_result(
            solver_name="cplex_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=status, labels=labels, bandwidth=bandwidth,
            lower_bound=lower, build_time=build_time, solve_time=solve_time,
            total_time=total_time,
            message="" if solution is not None else "no solution returned",
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return finalize_result(
            solver_name="cplex_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=elapsed, total_time=elapsed,
            message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if mdl is not None:
            try:
                mdl.end()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    from benchmark.runner import run_benchmark_suite

    parser = argparse.ArgumentParser(description="Run CPLEX benchmarks for CBMP")
    parser.add_argument(
        "--solver",
        choices=["cp", "mip", "both"],
        default="both",
        help="Which CPLEX solver to benchmark (default: both)",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit per instance")
    args = parser.parse_args()

    if args.solver in ("cp", "both"):
        run_benchmark_suite(benchmap_cplex_cp, solver_key="cplex_cp", time_limit=args.time_limit)

    if args.solver in ("mip", "both"):
        run_benchmark_suite(benchmap_cplex_mip, solver_key="cplex_mip", time_limit=args.time_limit)

