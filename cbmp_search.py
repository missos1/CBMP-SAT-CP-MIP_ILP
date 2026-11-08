"""Optimal bound search for the Cyclic Bandwidth Minimization Problem.

The SAT encodings decide ``CB(G) <= B``.  ``find_min_bandwidth`` searches for
the smallest feasible ``B`` (binary search, with optional incremental pairwise
variant) and always returns a valid ``[lower_bound, upper_bound]`` interval
together with the best labeling found.
"""

import argparse
import math
import multiprocessing
import queue as queue_module
import time
from dataclasses import dataclass, field
from typing import Optional

from pysat.solvers import Solver

from cbmp_cnf import (
    build_cnf,
    build_pairwise_cnf,
    decode_model,
    labels_at_distance,
    var_encode,
)
from classes.graph import Graph
from classes.verify import cyclic_bandwidth


SAT = "SAT"
UNSAT = "UNSAT"
TIMEOUT = "TIMEOUT"
ERROR = "ERROR"


@dataclass
class DecisionResult:
    bound: int
    status: str
    labels: Optional[dict] = None
    bandwidth: Optional[int] = None
    time: float = 0.0
    variables: int = 0
    clauses: int = 0
    message: str = ""


@dataclass
class SearchResult:
    graph_name: str
    vertex_count: int
    edge_count: int
    max_degree: int
    encoding: str
    solver: str
    lower_bound: int
    upper_bound: int
    optimal: bool
    bandwidth: Optional[int]
    labels: Optional[dict]
    status: str
    solve_calls: int
    time: float
    decisions: list = field(default_factory=list)


def max_degree(graph: Graph) -> int:
    return max(
        (len(neighbours) for neighbours in graph.adjacency_list.values()),
        default=0,
    )


def lower_bound(graph: Graph) -> int:
    """Theoretical lower bound ``ceil(Delta(G) / 2)`` (paper)."""
    if graph.get_vertex_count() <= 1 or graph.get_edge_count() == 0:
        return 0
    return max(1, math.ceil(max_degree(graph) / 2))


def upper_bound(graph: Graph) -> int:
    """Trivial upper bound ``floor(n / 2)`` (maximum cyclic distance)."""
    return graph.get_vertex_count() // 2


def _solve_inprocess(clauses, solver_name: str):
    try:
        with Solver(name=solver_name, bootstrap_with=clauses) as solver:
            answer = solver.solve()
            model = solver.get_model() if answer else None
        return (SAT if answer else UNSAT), model, ""
    except Exception as exc:
        return ERROR, None, f"{type(exc).__name__}: {exc}"


def _worker_solve(clauses, solver_name: str, result_queue) -> None:
    status, model, message = _solve_inprocess(clauses, solver_name)
    result_queue.put((status, model, message))


def _solve_subprocess(clauses, solver_name: str, time_limit: float):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_worker_solve, args=(clauses, solver_name, result_queue)
    )
    process.start()
    try:
        status, model, message = result_queue.get(timeout=max(time_limit, 0.0))
    except queue_module.Empty:
        status, model, message = TIMEOUT, None, "solver time limit reached"
    finally:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
    return status, model, message


def solve_decision(graph: Graph, bound: int, encoding: str = "pairwise",
                   solver_name: str = "cadical195",
                   time_limit: Optional[float] = None,
                   break_symmetry: bool = False,
                   break_reflection: bool = False,
                   amo_encoding="pairwise",
                   card_encoding="seqcounter") -> DecisionResult:
    start = time.perf_counter()
    try:
        cnf = build_cnf(
            graph, bound, encoding=encoding,
            break_symmetry=break_symmetry,
            break_reflection=break_reflection,
            amo_encoding=amo_encoding, card_encoding=card_encoding,
        )
    except Exception as exc:
        return DecisionResult(bound, ERROR, time=time.perf_counter() - start,
                              message=f"{type(exc).__name__}: {exc}")

    if time_limit is None:
        status, model, message = _solve_inprocess(cnf.clauses, solver_name)
    else:
        status, model, message = _solve_subprocess(
            cnf.clauses, solver_name, time_limit
        )

    elapsed = time.perf_counter() - start
    if status != SAT:
        return DecisionResult(
            bound, status, time=elapsed,
            variables=cnf.nv, clauses=len(cnf.clauses), message=message,
        )

    labels = decode_model(model, graph, encoding=encoding)
    return DecisionResult(
        bound, SAT, labels=labels,
        bandwidth=cyclic_bandwidth(graph, labels), time=elapsed,
        variables=cnf.nv, clauses=len(cnf.clauses),
    )


def _trivial_result(graph: Graph, encoding: str, solver_name: str,
                    start: float) -> SearchResult:
    labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
    return SearchResult(
        graph_name=graph.name,
        vertex_count=graph.get_vertex_count(),
        edge_count=graph.get_edge_count(),
        max_degree=max_degree(graph),
        encoding=encoding,
        solver=solver_name,
        lower_bound=0,
        upper_bound=0,
        optimal=True,
        bandwidth=0,
        labels=labels,
        status="TRIVIAL",
        solve_calls=0,
        time=time.perf_counter() - start,
    )


def find_min_bandwidth(graph: Graph, encoding: str = "pairwise",
                       solver_name: str = "cadical195",
                       time_limit_per_call: Optional[float] = None,
                       total_time_limit: Optional[float] = None,
                       break_symmetry: bool = False,
                       break_reflection: bool = False,
                       amo_encoding="pairwise",
                       card_encoding="seqcounter") -> SearchResult:
    """Binary search for the minimum feasible bound."""
    start = time.perf_counter()
    n = graph.get_vertex_count()

    if n <= 1 or graph.get_edge_count() == 0:
        return _trivial_result(graph, encoding, solver_name, start)

    witness = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
    best_labels = witness
    best_bandwidth = cyclic_bandwidth(graph, witness)

    lo = lower_bound(graph) - 1
    hi = best_bandwidth
    status = "OPTIMAL"
    message = ""
    decisions = []
    solve_calls = 0

    completed = True
    while hi - lo > 1:
        if total_time_limit is not None:
            remaining_total = total_time_limit - (time.perf_counter() - start)
            if remaining_total <= 0:
                status = TIMEOUT
                message = "total time limit reached"
                completed = False
                break
        else:
            remaining_total = None

        mid = (lo + hi) // 2
        budget = time_limit_per_call
        if remaining_total is not None:
            budget = remaining_total if budget is None else min(budget, remaining_total)

        result = solve_decision(
            graph, mid, encoding=encoding, solver_name=solver_name,
            time_limit=budget, break_symmetry=break_symmetry,
            break_reflection=break_reflection,
            amo_encoding=amo_encoding, card_encoding=card_encoding,
        )
        decisions.append(result)
        solve_calls += 1

        if result.status == SAT:
            hi = min(mid, result.bandwidth)
            best_labels = result.labels
            best_bandwidth = result.bandwidth
        elif result.status == UNSAT:
            lo = max(lo, mid)
        elif result.status == TIMEOUT:
            status = TIMEOUT
            message = result.message or "decision timed out"
            completed = False
            break
        else:
            status = ERROR
            message = result.message
            completed = False
            break

    optimal = hi == lo + 1
    if completed and not optimal:
        status = "FEASIBLE"

    return SearchResult(
        graph_name=graph.name,
        vertex_count=n,
        edge_count=graph.get_edge_count(),
        max_degree=max_degree(graph),
        encoding=encoding,
        solver=solver_name,
        lower_bound=lo + 1,
        upper_bound=hi,
        optimal=optimal,
        bandwidth=best_bandwidth,
        labels=best_labels,
        status=status,
        solve_calls=solve_calls,
        time=time.perf_counter() - start,
        decisions=decisions,
    )


def find_min_bandwidth_incremental(graph: Graph,
                                   solver_name: str = "cadical195",
                                   time_limit_per_call: Optional[float] = None,
                                   total_time_limit: Optional[float] = None,
                                   break_symmetry: bool = False,
                                   break_reflection: bool = False,
                                   amo_encoding="pairwise") -> SearchResult:
    """Incremental pairwise search: start from a feasible bound and tighten.

    The incremental formulation is solved in-process, so per-decision time
    limits are not enforced; ``total_time_limit`` is checked between decisions.
    """
    start = time.perf_counter()
    n = graph.get_vertex_count()
    if n <= 1 or graph.get_edge_count() == 0:
        return _trivial_result(graph, "pairwise", solver_name, start)

    vertices = list(graph.adjacency_list)
    index = {v: i for i, v in enumerate(vertices, 1)}
    witness = {v: i for i, v in enumerate(vertices, 1)}
    best_labels = witness
    best_bandwidth = cyclic_bandwidth(graph, witness)
    theoretical_lb = lower_bound(graph)

    cnf = build_pairwise_cnf(
        graph, best_bandwidth, break_symmetry, break_reflection, amo_encoding
    )

    status = "OPTIMAL"
    message = ""
    solve_calls = 0
    decisions = []
    bound = best_bandwidth

    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
        while bound > theoretical_lb:
            if total_time_limit is not None:
                if total_time_limit - (time.perf_counter() - start) <= 0:
                    status = TIMEOUT
                    message = "total time limit reached"
                    break

            for u, v in graph.get_edges():
                iu, iv = index[u], index[v]
                for label_u in range(1, n + 1):
                    for label_v in labels_at_distance(label_u, bound, n):
                        solver.add_clause([
                            -var_encode(label_u, iu, n),
                            -var_encode(label_v, iv, n),
                        ])

            call_start = time.perf_counter()
            try:
                answer = solver.solve()
                call_status = SAT if answer else UNSAT
                model = solver.get_model() if answer else None
            except Exception as exc:
                status = ERROR
                message = f"{type(exc).__name__}: {exc}"
                break
            solve_calls += 1

            if call_status == UNSAT:
                decisions.append(DecisionResult(
                    bound - 1, UNSAT, time=time.perf_counter() - call_start,
                ))
                return SearchResult(
                    graph_name=graph.name,
                    vertex_count=n,
                    edge_count=graph.get_edge_count(),
                    max_degree=max_degree(graph),
                    encoding="pairwise",
                    solver=solver_name,
                    lower_bound=bound,
                    upper_bound=bound,
                    optimal=True,
                    bandwidth=bound,
                    labels=best_labels,
                    status="OPTIMAL",
                    solve_calls=solve_calls,
                    time=time.perf_counter() - start,
                    decisions=decisions,
                )
            if call_status == ERROR:
                status = ERROR
                message = "solver error"
                break

            labels = decode_model(model, graph, encoding="pairwise")
            actual = cyclic_bandwidth(graph, labels)
            decisions.append(DecisionResult(
                bound - 1, SAT, labels=labels, bandwidth=actual,
                time=time.perf_counter() - call_start,
            ))
            if actual < best_bandwidth:
                best_bandwidth = actual
                best_labels = labels
            bound -= 1

    return SearchResult(
        graph_name=graph.name,
        vertex_count=n,
        edge_count=graph.get_edge_count(),
        max_degree=max_degree(graph),
        encoding="pairwise",
        solver=solver_name,
        lower_bound=theoretical_lb,
        upper_bound=best_bandwidth,
        optimal=(status == "OPTIMAL" and best_bandwidth <= theoretical_lb),
        bandwidth=best_bandwidth,
        labels=best_labels,
        status=status,
        solve_calls=solve_calls,
        time=time.perf_counter() - start,
        decisions=decisions,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Solve CBMP for one matrix.")
    parser.add_argument("path", help="matrix file (.mtx/.mtx.gz/.pse.gz)")
    parser.add_argument("--encoding", choices=("pairwise", "order"),
                        default="pairwise")
    parser.add_argument("--solver", default="cadical195")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="per-decision time limit in seconds")
    parser.add_argument("--total-time-limit", type=float, default=None)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--symmetry", action="store_true")
    parser.add_argument("--reflection", action="store_true")
    args = parser.parse_args(argv)

    from mtx_parser import parse_file

    name, n, edges = parse_file(args.path)
    graph = Graph.from_parsed((name, n, edges))

    common = dict(
        solver_name=args.solver,
        time_limit_per_call=args.time_limit,
        total_time_limit=args.total_time_limit,
        break_symmetry=args.symmetry,
        break_reflection=args.reflection,
    )
    if args.incremental:
        result = find_min_bandwidth_incremental(graph, **common)
    else:
        result = find_min_bandwidth(graph, encoding=args.encoding, **common)

    print(f"graph={result.graph_name} n={result.vertex_count} "
          f"m={result.edge_count} delta={result.max_degree} "
          f"encoding={result.encoding} solver={result.solver}")
    print(f"status={result.status} optimal={result.optimal} "
          f"CB in [{result.lower_bound}, {result.upper_bound}] "
          f"best={result.bandwidth} calls={result.solve_calls} "
          f"time={result.time:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
