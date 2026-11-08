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
    effective_total_limit = time_limit if total_time_limit is None else total_time_limit
    if incremental:
        result = find_min_bandwidth_incremental(
            graph, solver_name=solver_name,
            time_limit_per_call=time_limit,
            total_time_limit=effective_total_limit,
            break_symmetry=break_symmetry,
            break_reflection=break_reflection,
            amo_encoding=amo_encoding,
        )
        encoding_name = "pairwise_incremental"
    else:
        result = find_min_bandwidth(
            graph, encoding=encoding, solver_name=solver_name,
            time_limit_per_call=time_limit,
            total_time_limit=effective_total_limit,
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
        build_time=result.build_time, solve_time=result.solve_time,
        extra={"solve_calls": result.solve_calls, "sat_status": result.status},
    )


if __name__ == "__main__":
    import argparse
    from functools import partial
    from benchmark.runner import run_benchmark_suite

    parser = argparse.ArgumentParser(description="Run PySAT benchmarks for CBMP")
    parser.add_argument(
        "--mode",
        choices=["pairwise", "incremental", "order", "all"],
        default="all",
        help="Which SAT configuration to benchmark (default: all)",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit per instance")
    args = parser.parse_args()

    if args.mode in ("pairwise", "all"):
        fn = partial(benchmap_sat, encoding="pairwise", incremental=False)
        run_benchmark_suite(
            fn, solver_key="sat_pairwise", encoding="pairwise", time_limit=args.time_limit
        )

    if args.mode in ("incremental", "all"):
        fn = partial(benchmap_sat, encoding="pairwise", incremental=True)
        run_benchmark_suite(
            fn,
            solver_key="sat_incremental",
            encoding="pairwise_incremental",
            time_limit=args.time_limit,
        )

    if args.mode in ("order", "all"):
        fn = partial(benchmap_sat, encoding="order", incremental=False)
        run_benchmark_suite(
            fn, solver_key="sat_order", encoding="order", time_limit=args.time_limit
        )

