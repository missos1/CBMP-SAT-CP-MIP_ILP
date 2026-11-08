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
                    message: str = "", extra=None) -> BenchmarkResult:
    verified = None
    if labels is not None:
        verified, actual, error = verify_labeling(graph, labels)
        if verified:
            bandwidth = actual
            if upper_bound is None or actual < upper_bound:
                upper_bound = actual
        else:
            message = f"{message} {error}".strip()

    if bandwidth is not None and (upper_bound is None or bandwidth < upper_bound):
        upper_bound = bandwidth

    return BenchmarkResult(
        solver_name=solver_name,
        graph_name=graph.name,
        vertex_count=graph.get_vertex_count(),
        edge_count=graph.get_edge_count(),
        symmetry_breaking=symmetry_breaking,
        reflection_breaking=reflection_breaking,
        encoding=encoding,
        build_time=round(build_time, 6),
        solve_time=round(solve_time, 6),
        total_time=round(build_time + solve_time, 6),
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
