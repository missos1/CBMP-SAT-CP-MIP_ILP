"""SAT search wrapped as a benchmark baseline."""

from benchmark.common import (
    ERROR,
    FEASIBLE,
    OPTIMAL,
    TIMEOUT,
    UNKNOWN,
    finalize_result,
)
from cbmp_search import find_min_bandwidth, find_min_bandwidth_incremental
from classes.graph import Graph


def benchmap_sat(graph: Graph, encoding: str = "pairwise",
                 solver_name: str = "cadical195",
                 time_limit: float = 60.0,
                 total_time_limit=None,
                 break_symmetry: bool = False,
                 break_reflection: bool = False,
                 incremental: bool = False,
                 amo_encoding="pairwise",
                 card_encoding="seqcounter") -> object:
    if incremental:
        result = find_min_bandwidth_incremental(
            graph, solver_name=solver_name,
            time_limit_per_call=time_limit,
            total_time_limit=total_time_limit,
            break_symmetry=break_symmetry,
            break_reflection=break_reflection,
            amo_encoding=amo_encoding,
        )
        encoding_name = "pairwise_incremental"
    else:
        result = find_min_bandwidth(
            graph, encoding=encoding, solver_name=solver_name,
            time_limit_per_call=time_limit,
            total_time_limit=total_time_limit,
            break_symmetry=break_symmetry,
            break_reflection=break_reflection,
            amo_encoding=amo_encoding, card_encoding=card_encoding,
        )
        encoding_name = encoding

    if result.status in ("OPTIMAL", "TRIVIAL"):
        status = OPTIMAL
    elif result.status == "TIMEOUT":
        status = TIMEOUT
    elif result.status == "FEASIBLE":
        status = FEASIBLE
    elif result.status == "ERROR":
        status = ERROR
    else:
        status = UNKNOWN

    return finalize_result(
        solver_name="sat", graph=graph,
        symmetry_breaking=break_symmetry,
        reflection_breaking=break_reflection,
        encoding=encoding_name, status=status,
        labels=result.labels, bandwidth=result.bandwidth,
        lower_bound=result.lower_bound, upper_bound=result.upper_bound,
        solve_time=result.time,
        extra={"solve_calls": result.solve_calls, "sat_status": result.status},
    )
