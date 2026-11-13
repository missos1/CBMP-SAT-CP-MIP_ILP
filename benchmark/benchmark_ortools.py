"""OR-Tools CP-SAT baseline for CBMP."""

import math
import time

from ortools.sat.python import cp_model

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


def _map_status(status: int, time_limit: float, total_elapsed: float) -> str:
    if status == cp_model.OPTIMAL:
        return OPTIMAL
    if status == cp_model.FEASIBLE:
        return FEASIBLE
    if status == cp_model.INFEASIBLE:
        return INFEASIBLE
    if status == cp_model.UNKNOWN:
        return TIMEOUT
    return ERROR


def benchmap_cpsat(graph: Graph, break_symmetry: bool = False,
                   break_reflection: bool = False,
                   time_limit: float = 60.0,
                   workers: int = 8) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="cpsat", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    theo_lb = theoretical_lower_bound(graph)
    start = time.perf_counter()

    try:
        model = cp_model.CpModel()
        position = {v: model.NewIntVar(0, n - 1, f"p_{i}")
                    for i, v in enumerate(vertices)}
        model.AddAllDifferent(list(position.values()))

        if break_symmetry:
            model.Add(position[vertices[0]] == 0)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = neighbours[0] if neighbours else vertices[1]
                model.Add(position[target] <= n // 2)

        distances = []
        for u, v in edges:
            signed = position[v] - position[u] + n
            offset = model.NewIntVar(0, n - 1, "offset")
            model.AddModuloEquality(offset, signed, n)
            distance = model.NewIntVar(0, n // 2, "dist")
            model.AddMinEquality(distance, [offset, n - offset])
            distances.append(distance)

        bound = model.NewIntVar(theo_lb, n // 2, "B")
        model.AddMaxEquality(bound, distances)
        model.Minimize(bound)

        build_time = time.perf_counter() - start
        remaining = max(0.0, float(time_limit) - build_time)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = int(workers)

        solve_start = time.perf_counter()
        status = solver.Solve(model) if remaining > 0 else cp_model.UNKNOWN
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start

        mapped = _map_status(status, time_limit, total_time)
        labels = None
        bandwidth = None
        if remaining > 0 and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            labels = {
                key: int(solver.Value(position[key])) + 1 for key in vertices
            }
            bandwidth = int(round(solver.ObjectiveValue()))

        lower = theo_lb
        if remaining > 0:
            try:
                best_bound = float(solver.BestObjectiveBound())
                if math.isfinite(best_bound) and abs(best_bound) < 1e20:
                    lower = max(lower, int(math.ceil(best_bound - 1e-6)))
            except Exception:
                pass
        if lower > n // 2:
            lower = n // 2

        return finalize_result(
            solver_name="cpsat", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=mapped, labels=labels, bandwidth=bandwidth,
            lower_bound=lower, build_time=build_time, solve_time=solve_time,
            total_time=total_time,
            extra={"cpsat_status": int(status)},
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return finalize_result(
            solver_name="cpsat", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=elapsed, total_time=elapsed,
            message=f"{type(exc).__name__}: {exc}",
        )


benchmap_ortools = benchmap_cpsat


if __name__ == "__main__":
    import argparse
    from benchmark.runner import run_benchmark_suite

    parser = argparse.ArgumentParser(description="Run OR-Tools CP-SAT benchmark for CBMP")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit per instance")
    args = parser.parse_args()

    run_benchmark_suite(benchmap_cpsat, solver_key="cpsat", time_limit=args.time_limit)

