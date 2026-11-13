"""Shared helpers for exact solver baselines."""

import math

from classes.benchmarkresult import BenchmarkResult
from classes.graph import Graph
from classes.verify import verify_labeling


OPTIMAL = "OPTIMAL"
FEASIBLE = "FEASIBLE"
TIMEOUT = "TIMEOUT"
INFEASIBLE = "INFEASIBLE"
UNKNOWN = "UNKNOWN"
ERROR = "ERROR"


def theoretical_lower_bound(graph: Graph) -> int:
    if graph.get_vertex_count() <= 1 or graph.get_edge_count() == 0:
        return 0
    delta = max(
        (len(neighbours) for neighbours in graph.adjacency_list.values()),
        default=0,
    )
    return max(1, math.ceil(delta / 2))


def finalize_result(*, solver_name: str, graph: Graph,
                    symmetry_breaking: bool, reflection_breaking: bool,
                    status: str, encoding=None, labels=None,
                    bandwidth=None, lower_bound=None, upper_bound=None,
                    build_time: float = 0.0, solve_time: float = 0.0,
                    total_time: float = None,
                    message: str = "", extra=None) -> BenchmarkResult:
    verified = None
    if labels is not None:
        verified, actual, error = verify_labeling(graph, labels)
        if verified:
            bandwidth = actual
            upper_bound = actual
            if status in (TIMEOUT, UNKNOWN):
                status = FEASIBLE
        else:
            message = f"{message} {error}".strip()
            labels = None
            bandwidth = None
            upper_bound = None
            if status in (OPTIMAL, FEASIBLE):
                status = ERROR
    else:
        bandwidth = None
        upper_bound = None
        if status in (OPTIMAL, FEASIBLE):
            status = TIMEOUT

    if verified and bandwidth is not None:
        if lower_bound is not None and bandwidth <= lower_bound:
            status = OPTIMAL
        if status == OPTIMAL:
            lower_bound = bandwidth
            upper_bound = bandwidth

    build_val = max(0.0, float(build_time or 0.0))
    solve_val = max(0.0, float(solve_time or 0.0))
    total_val = max(0.0, float(total_time)) if total_time is not None else (build_val + solve_val)

    return BenchmarkResult(
        solver_name=solver_name,
        graph_name=graph.name,
        vertex_count=graph.get_vertex_count(),
        edge_count=graph.get_edge_count(),
        symmetry_breaking=symmetry_breaking,
        reflection_breaking=reflection_breaking,
        encoding=encoding,
        build_time=round(build_val, 6),
        solve_time=round(solve_val, 6),
        total_time=round(total_val, 6),
        bandwidth=bandwidth,
        status=status,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        optimal=(status == OPTIMAL),
        solution=labels,
        verified=verified,
        message=message,
        extra=dict(extra or {}),
    )
