"""HiGHS MIP baseline for CBMP (compact assignment formulation)."""

import math
import time

import highspy

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


FEASIBLE_SOLUTION = 2  # HighsInfo.primal_solution_status for a feasible point


def _qsum(terms):
    return highspy.qsum(terms) if hasattr(highspy, "qsum") else sum(terms)


def _map_status(model_status, has_solution: bool) -> str:
    status = highspy.HighsModelStatus
    if model_status == status.kOptimal:
        return OPTIMAL
    if model_status == status.kInfeasible:
        return INFEASIBLE
    if model_status in (status.kTimeLimit, status.kInterrupt,
                        status.kIterationLimit, status.kSolutionLimit):
        return FEASIBLE if has_solution else TIMEOUT
    return FEASIBLE if has_solution else UNKNOWN


def benchmap_highs(graph: Graph, break_symmetry: bool = False,
                   break_reflection: bool = False,
                   time_limit: float = 60.0,
                   mip_gap: float = 0.0,
                   workers: int = 8) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="highs_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    index = {v: i for i, v in enumerate(vertices)}
    theo_lb = theoretical_lower_bound(graph)
    start = time.perf_counter()

    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("parallel", "on")
        h.setOptionValue("threads", int(workers))
        h.setOptionValue("mip_rel_gap", float(mip_gap))
        h.setOptionValue("mip_abs_gap", 0.0)

        x = [[h.addBinary(name=f"x_{i}_{l}") for l in range(n)] for i in range(n)]
        for i in range(n):
            h.addConstr(_qsum(x[i][l] for l in range(n)) == 1)
        for l in range(n):
            h.addConstr(_qsum(x[i][l] for i in range(n)) == 1)

        if break_symmetry:
            h.addConstr(x[0][0] == 1)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = index[neighbours[0]] if neighbours else 1
                h.addConstr(_qsum(x[target][l] for l in range(n // 2 + 1, n)) == 0)

        bound = h.addIntegral(theo_lb, n // 2, name="B")
        position = [h.addIntegral(0, n - 1, name=f"p_{i}") for i in range(n)]
        for i in range(n):
            h.addConstr(position[i] - _qsum(l * x[i][l] for l in range(n)) == 0)

        for edge_index, (u, v) in enumerate(edges):
            iu, iv = index[u], index[v]
            offset = h.addIntegral(0, n - 1, name=f"d_{edge_index}")
            wrap = h.addBinary(name=f"w_{edge_index}")
            side = h.addBinary(name=f"s_{edge_index}")
            h.addConstr(position[iv] - position[iu] - offset + n * wrap == 0)
            h.addConstr(offset - bound + n * side <= n)
            h.addConstr(n - offset - bound - n * side <= 0)

        build_time = time.perf_counter() - start
        remaining = max(0.0, float(time_limit) - build_time)
        h.setOptionValue("time_limit", remaining)

        solve_start = time.perf_counter()
        if remaining > 0:
            h.minimize(bound)
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start

        info = h.getInfo()
        has_solution = (remaining > 0 and info.primal_solution_status == FEASIBLE_SOLUTION)
        status = _map_status(h.getModelStatus(), has_solution) if remaining > 0 else TIMEOUT

        labels = None
        upper = None
        if has_solution:
            values = h.getSolution().col_value
            labels = {}
            for i, key in enumerate(vertices):
                best_l = max(range(n), key=lambda l: values[x[i][l].index])
                if values[x[i][best_l].index] > 0.5:
                    labels[key] = best_l + 1
            upper = int(round(info.objective_function_value))

        lower = theo_lb
        dual = info.mip_dual_bound if remaining > 0 else float("nan")
        if math.isfinite(dual) and abs(dual) < 1e20:
            lower = max(lower, int(math.ceil(dual - 1e-6)))
        if lower > n // 2:
            lower = n // 2

        return finalize_result(
            solver_name="highs_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=status, labels=labels,
            lower_bound=lower, upper_bound=upper,
            build_time=build_time, solve_time=solve_time, total_time=total_time,
            extra={
                "highs_status": h.modelStatusToString(h.getModelStatus()) if remaining > 0 else "Time limit reached",
                "mip_gap": float(info.mip_gap) if remaining > 0 else float("inf"),
                "mip_node_count": int(info.mip_node_count) if remaining > 0 else 0,
            },
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return finalize_result(
            solver_name="highs_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=elapsed, total_time=elapsed,
            message=f"{type(exc).__name__}: {exc}",
        )


benchmap_highs_mip = benchmap_highs


if __name__ == "__main__":
    import argparse
    from benchmark.runner import run_benchmark_suite

    parser = argparse.ArgumentParser(description="Run HiGHS MIP benchmark for CBMP")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit per instance")
    args = parser.parse_args()

    run_benchmark_suite(benchmap_highs, solver_key="highs_mip", time_limit=args.time_limit)

