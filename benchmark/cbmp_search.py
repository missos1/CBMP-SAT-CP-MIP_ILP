"""Optimal bound search for the Cyclic Bandwidth Minimization Problem.

The SAT encodings decide ``CB(G) <= B``.  ``find_min_bandwidth`` searches for
the smallest feasible ``B`` (binary search, with optional incremental pairwise
variant) and always returns a valid ``[lower_bound, upper_bound]`` interval
together with the best labeling found.
"""

import argparse
import math
import multiprocessing as mp
import os
import queue as queue_module
import time
from dataclasses import dataclass, field
from typing import Optional

from pysat.solvers import Solver

from benchmark.cbmp_cnf import (
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
    build_time: float = 0.0
    solve_time: float = 0.0


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
    build_time: float = 0.0
    solve_time: float = 0.0


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


def _mp_context():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


def _terminate_process(process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=0.5)
    else:
        process.join(timeout=0.5)
    try:
        if process.pid:
            os.waitpid(process.pid, os.WNOHANG)
    except OSError:
        pass


def _solve_inprocess(clauses, solver_name: str):
    """Solve ``clauses``; returns (status, model, message, solve_seconds)."""
    try:
        solve_start = time.perf_counter()
        with Solver(name=solver_name, bootstrap_with=clauses) as solver:
            answer = solver.solve()
            model = solver.get_model() if answer else None
        solve_seconds = time.perf_counter() - solve_start
        return (SAT if answer else UNSAT), model, "", solve_seconds
    except Exception as exc:
        return ERROR, None, f"{type(exc).__name__}: {exc}", 0.0


def _decision_worker(graph: Graph, bound: int, encoding: str, solver_name: str,
                     break_symmetry: bool, break_reflection: bool,
                     amo_encoding: str, card_encoding: str, send_conn) -> None:
    try:
        t0 = time.perf_counter()
        cnf = build_cnf(
            graph, bound, encoding=encoding,
            break_symmetry=break_symmetry,
            break_reflection=break_reflection,
            amo_encoding=amo_encoding, card_encoding=card_encoding,
        )
        build_time = time.perf_counter() - t0
        send_conn.send(("BUILT", build_time, cnf.nv, len(cnf.clauses)))

        t1 = time.perf_counter()
        with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
            answer = solver.solve()
            model = solver.get_model() if answer else None
        if answer:
            labels = decode_model(model, graph, encoding=encoding)
            bw = cyclic_bandwidth(graph, labels)
            status = SAT
        else:
            labels = None
            bw = None
            status = UNSAT
        solve_time = time.perf_counter() - t1
        send_conn.send(("DONE", status, labels, bw, "", solve_time))
    except Exception as exc:
        send_conn.send(("ERROR", f"{type(exc).__name__}: {exc}"))
    finally:
        send_conn.close()


def solve_decision(graph: Graph, bound: int, encoding: str = "pairwise",
                   solver_name: str = "cadical195",
                   time_limit: Optional[float] = None,
                   break_symmetry: bool = False,
                   break_reflection: bool = False,
                   amo_encoding="pairwise",
                   card_encoding="seqcounter") -> DecisionResult:
    start = time.perf_counter()
    if time_limit is None:
        try:
            cnf = build_cnf(
                graph, bound, encoding=encoding,
                break_symmetry=break_symmetry,
                break_reflection=break_reflection,
                amo_encoding=amo_encoding, card_encoding=card_encoding,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return DecisionResult(bound, ERROR, time=elapsed, build_time=elapsed,
                                  message=f"{type(exc).__name__}: {exc}")
        build_time = time.perf_counter() - start
        status, model, message, _ = _solve_inprocess(cnf.clauses, solver_name)
        if status == SAT:
            labels = decode_model(model, graph, encoding=encoding)
            bw = cyclic_bandwidth(graph, labels)
        else:
            labels = None
            bw = None
        elapsed = time.perf_counter() - start
        solve_time = max(0.0, elapsed - build_time)
        return DecisionResult(
            bound, status, labels=labels, bandwidth=bw, time=elapsed,
            variables=cnf.nv, clauses=len(cnf.clauses), message=message,
            build_time=build_time, solve_time=solve_time,
        )

    if time_limit <= 0:
        return DecisionResult(
            bound, TIMEOUT, time=0.0, build_time=0.0, solve_time=0.0,
            message="solver time limit reached",
        )

    ctx = _mp_context()
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_decision_worker,
        args=(
            graph, bound, encoding, solver_name,
            break_symmetry, break_reflection,
            amo_encoding, card_encoding, send_conn,
        ),
    )
    process.start()
    send_conn.close()

    deadline = start + float(time_limit)
    built = False
    build_time = 0.0
    variables = 0
    clauses = 0
    status = TIMEOUT
    labels = None
    bw = None
    message = "solver time limit reached"

    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                status = TIMEOUT
                message = "solver time limit reached"
                break
            step_timeout = min(1.0, max(0.01, remaining))
            if recv_conn.poll(step_timeout):
                try:
                    msg = recv_conn.recv()
                except EOFError:
                    break
                tag = msg[0]
                if tag == "BUILT":
                    _, build_time, variables, clauses = msg
                    built = True
                elif tag == "DONE":
                    _, status, labels, bw, message, _ = msg
                    break
                elif tag == "ERROR":
                    _, message = msg
                    status = ERROR
                    break
            elif not process.is_alive():
                if not built:
                    status = ERROR
                    message = "worker process terminated during CNF build"
                else:
                    status = TIMEOUT
                    message = "solver time limit reached"
                break
    finally:
        _terminate_process(process)
        recv_conn.close()

    elapsed = time.perf_counter() - start
    if built:
        build_time = min(build_time, elapsed)
        solve_time = max(0.0, elapsed - build_time)
    else:
        build_time = elapsed
        solve_time = 0.0

    return DecisionResult(
        bound, status, labels=labels, bandwidth=bw, time=elapsed,
        variables=variables, clauses=clauses, message=message,
        build_time=build_time, solve_time=solve_time,
    )


def _trivial_result(graph: Graph, encoding: str, solver_name: str,
                    start: float) -> SearchResult:
    labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
    elapsed = time.perf_counter() - start
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
        time=elapsed,
        build_time=elapsed,
        solve_time=0.0,
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

    theo_lb = lower_bound(graph)
    theo_ub = upper_bound(graph)
    lo = theo_lb - 1
    hi = theo_ub
    best_labels = None
    best_bandwidth = None
    decisions = []
    solve_calls = 0
    timed_out = False
    had_error = False
    message = ""

    while hi - lo > 1 or best_labels is None:
        if total_time_limit is not None:
            remaining_total = total_time_limit - (time.perf_counter() - start)
            if remaining_total <= 0:
                timed_out = True
                message = "total time limit reached"
                break
        else:
            remaining_total = None

        mid = (lo + hi) // 2 if hi - lo > 1 else hi
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
            best_labels = result.labels
            best_bandwidth = result.bandwidth
            hi = min(mid, result.bandwidth)
            if hi <= lo + 1:
                break
        elif result.status == UNSAT:
            lo = max(lo, mid)
            if lo >= hi:
                break
        elif result.status == TIMEOUT:
            timed_out = True
            message = result.message or "decision timed out"
            break
        else:
            had_error = True
            message = result.message
            break

    optimal = (not timed_out and not had_error and best_labels is not None and hi <= lo + 1)
    if optimal:
        status = "OPTIMAL"
        final_lb = best_bandwidth
        final_ub = best_bandwidth
    elif best_labels is not None:
        status = "FEASIBLE"
        final_lb = lo + 1
        final_ub = best_bandwidth
    elif had_error:
        status = ERROR
        final_lb = lo + 1
        final_ub = None
    else:
        status = TIMEOUT
        final_lb = lo + 1
        final_ub = None

    total_wall = time.perf_counter() - start
    total_build = min(total_wall, sum(d.build_time for d in decisions))
    total_solve = max(0.0, total_wall - total_build)

    return SearchResult(
        graph_name=graph.name,
        vertex_count=n,
        edge_count=graph.get_edge_count(),
        max_degree=max_degree(graph),
        encoding=encoding,
        solver=solver_name,
        lower_bound=final_lb,
        upper_bound=final_ub,
        optimal=optimal,
        bandwidth=best_bandwidth,
        labels=best_labels,
        status=status,
        solve_calls=solve_calls,
        time=total_wall,
        decisions=decisions,
        build_time=total_build,
        solve_time=total_solve,
    )


def _incremental_worker(graph: Graph, solver_name: str,
                        break_symmetry: bool, break_reflection: bool,
                        amo_encoding: str, send_conn) -> None:
    try:
        n = graph.get_vertex_count()
        vertices = list(graph.adjacency_list)
        index = {v: i for i, v in enumerate(vertices, 1)}
        theo_lb = lower_bound(graph)
        theo_ub = upper_bound(graph)

        t0 = time.perf_counter()
        cnf = build_pairwise_cnf(
            graph, theo_ub, break_symmetry, break_reflection, amo_encoding
        )
        init_build_time = time.perf_counter() - t0
        send_conn.send(("BUILD", init_build_time))

        with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
            enforced_bound = theo_ub
            best_bw = None

            while True:
                if best_bw is not None and best_bw <= theo_lb:
                    send_conn.send(("OPTIMAL_REACHED",))
                    break

                if best_bw is not None:
                    target = best_bw - 1
                    t_add = time.perf_counter()
                    for d in range(enforced_bound, target, -1):
                        for u, v in graph.get_edges():
                            iu, iv = index[u], index[v]
                            for label_u in range(1, n + 1):
                                for label_v in labels_at_distance(label_u, d, n):
                                    solver.add_clause([
                                        -var_encode(label_u, iu, n),
                                        -var_encode(label_v, iv, n),
                                    ])
                    add_time = time.perf_counter() - t_add
                    enforced_bound = target
                    send_conn.send(("BUILD", add_time))

                t_solve = time.perf_counter()
                send_conn.send(("SOLVE_START", enforced_bound))
                answer = solver.solve()
                model = solver.get_model() if answer else None
                call_time = time.perf_counter() - t_solve

                if not answer:
                    send_conn.send(("UNSAT", enforced_bound, call_time))
                    break

                labels = decode_model(model, graph, encoding="pairwise")
                actual = cyclic_bandwidth(graph, labels)
                if best_bw is None or actual < best_bw:
                    best_bw = actual
                else:
                    best_bw = min(best_bw, enforced_bound)
                send_conn.send(("SAT", enforced_bound, labels, actual, call_time))
    except Exception as exc:
        send_conn.send(("ERROR", f"{type(exc).__name__}: {exc}"))
    finally:
        send_conn.close()


def find_min_bandwidth_incremental(graph: Graph,
                                   solver_name: str = "cadical195",
                                   time_limit_per_call: Optional[float] = None,
                                   total_time_limit: Optional[float] = None,
                                   break_symmetry: bool = False,
                                   break_reflection: bool = False,
                                   amo_encoding="pairwise") -> SearchResult:
    """Incremental pairwise search: start from upper bound and tighten in a subprocess."""
    start = time.perf_counter()
    n = graph.get_vertex_count()
    if n <= 1 or graph.get_edge_count() == 0:
        return _trivial_result(graph, "pairwise", solver_name, start)

    theo_lb = lower_bound(graph)
    theo_ub = upper_bound(graph)
    best_labels = None
    best_bandwidth = None
    build_time = 0.0
    solve_calls = 0
    decisions = []
    timed_out = False
    had_error = False
    is_optimal = False
    message = ""

    ctx = _mp_context()
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_incremental_worker,
        args=(
            graph, solver_name,
            break_symmetry, break_reflection,
            amo_encoding, send_conn,
        ),
    )
    process.start()
    send_conn.close()

    current_call_start = None
    current_bound = theo_ub
    init_built = False

    try:
        while True:
            now = time.perf_counter()
            wait_timeout = None
            if total_time_limit is not None:
                rem_total = total_time_limit - (now - start)
                if rem_total <= 0:
                    timed_out = True
                    message = "total time limit reached"
                    break
                wait_timeout = rem_total
            if time_limit_per_call is not None and current_call_start is not None:
                rem_call = time_limit_per_call - (now - current_call_start)
                if rem_call <= 0:
                    timed_out = True
                    message = "decision timed out"
                    break
                wait_timeout = rem_call if wait_timeout is None else min(wait_timeout, rem_call)

            step_timeout = 1.0 if wait_timeout is None else min(1.0, max(0.01, wait_timeout))
            if recv_conn.poll(step_timeout):
                try:
                    msg = recv_conn.recv()
                except EOFError:
                    break
            elif wait_timeout is not None and wait_timeout <= 1.0:
                timed_out = True
                message = "solver time limit reached"
                break
            elif not process.is_alive():
                timed_out = True
                message = "worker process terminated"
                break
            else:
                continue

            tag = msg[0]
            if tag == "BUILD":
                _, dt = msg
                build_time += dt
                init_built = True
            elif tag == "SOLVE_START":
                _, current_bound = msg
                current_call_start = time.perf_counter()
            elif tag == "SAT":
                _, b, labels, actual, dt = msg
                current_call_start = None
                solve_calls += 1
                decisions.append(DecisionResult(
                    b, SAT, labels=labels, bandwidth=actual,
                    time=dt, solve_time=dt,
                ))
                if best_bandwidth is None or actual < best_bandwidth:
                    best_bandwidth = actual
                    best_labels = labels
            elif tag == "UNSAT":
                _, b, dt = msg
                current_call_start = None
                solve_calls += 1
                decisions.append(DecisionResult(
                    b, UNSAT, time=dt, solve_time=dt,
                ))
                is_optimal = (best_labels is not None)
                break
            elif tag == "OPTIMAL_REACHED":
                current_call_start = None
                is_optimal = (best_labels is not None)
                break
            elif tag == "ERROR":
                _, message = msg
                current_call_start = None
                had_error = True
                break
    finally:
        if timed_out and current_call_start is not None:
            call_elapsed = max(0.0, time.perf_counter() - current_call_start)
            solve_calls += 1
            decisions.append(DecisionResult(
                current_bound, TIMEOUT, time=call_elapsed, solve_time=call_elapsed,
                message=message or "solver time limit reached",
            ))
        _terminate_process(process)
        recv_conn.close()

    total_wall = time.perf_counter() - start
    if init_built:
        total_build = min(total_wall, build_time)
        total_solve = max(0.0, total_wall - total_build)
    else:
        total_build = total_wall
        total_solve = 0.0

    if is_optimal and not timed_out and not had_error and best_labels is not None:
        status = "OPTIMAL"
        optimal = True
        final_lb = best_bandwidth
        final_ub = best_bandwidth
    elif best_labels is not None:
        status = "FEASIBLE"
        optimal = False
        final_lb = theo_lb
        final_ub = best_bandwidth
    elif had_error:
        status = ERROR
        optimal = False
        final_lb = theo_lb
        final_ub = None
    else:
        status = TIMEOUT
        optimal = False
        final_lb = theo_lb
        final_ub = None

    return SearchResult(
        graph_name=graph.name,
        vertex_count=n,
        edge_count=graph.get_edge_count(),
        max_degree=max_degree(graph),
        encoding="pairwise",
        solver=solver_name,
        lower_bound=final_lb,
        upper_bound=final_ub,
        optimal=optimal,
        bandwidth=best_bandwidth,
        labels=best_labels,
        status=status,
        solve_calls=solve_calls,
        time=total_wall,
        decisions=decisions,
        build_time=total_build,
        solve_time=total_solve,
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

    from benchmark.mtx_parser import parse_file

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
