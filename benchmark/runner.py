"""Unified benchmark runner with live terminal status ticker and crash-resilient CSV saving."""

import csv
import json
import os
import sys
import threading
import time
from typing import Callable, List, Optional

from benchmark.config import CONFIG_LIST, GRAPH_LIST, TIME_LIMIT
from classes.benchmarkresult import BenchmarkResult
from classes.graph import Graph


CSV_COLUMNS = [
    "solver_name",
    "graph_name",
    "vertex_count",
    "edge_count",
    "symmetry_breaking",
    "reflection_breaking",
    "encoding",
    "build_time",
    "solve_time",
    "total_time",
    "bandwidth",
    "status",
    "lower_bound",
    "upper_bound",
    "optimal",
    "verified",
    "message",
    "extra",
]


def _result_to_row(result: BenchmarkResult) -> dict:
    return {
        "solver_name": result.solver_name,
        "graph_name": result.graph_name,
        "vertex_count": result.vertex_count,
        "edge_count": result.edge_count,
        "symmetry_breaking": result.symmetry_breaking,
        "reflection_breaking": result.reflection_breaking,
        "encoding": result.encoding or "",
        "build_time": result.build_time,
        "solve_time": result.solve_time,
        "total_time": result.total_time,
        "bandwidth": result.bandwidth if result.bandwidth is not None else "",
        "status": result.status,
        "lower_bound": result.lower_bound if result.lower_bound is not None else "",
        "upper_bound": result.upper_bound if result.upper_bound is not None else "",
        "optimal": result.optimal,
        "verified": result.verified if result.verified is not None else "",
        "message": result.message or "",
        "extra": json.dumps(result.extra) if result.extra else "",
    }


def load_completed_keys(csv_path: str) -> set:
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ERROR":
                    continue
                g_name = row.get("graph_name", "")
                sym = row.get("symmetry_breaking", "False").lower() == "true"
                refl = row.get("reflection_breaking", "False").lower() == "true"
                enc = row.get("encoding", "")
                completed.add((g_name, sym, refl, enc))
    except Exception:
        pass
    return completed


def append_result(csv_path: str, result: BenchmarkResult) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    row = _result_to_row(result)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def run_benchmark_suite(
    solver_fn: Callable[..., BenchmarkResult],
    solver_key: str,
    output_csv: Optional[str] = None,
    graphs: Optional[List[Graph]] = None,
    configs: Optional[List[tuple]] = None,
    time_limit: Optional[float] = None,
    encoding: Optional[str] = None,
    extra_kwargs: Optional[dict] = None,
) -> None:
    graphs = graphs or GRAPH_LIST
    configs = configs or CONFIG_LIST
    time_limit = time_limit or TIME_LIMIT
    extra_kwargs = extra_kwargs or {}

    if output_csv is None:
        output_csv = os.path.join("result_csv", f"{solver_key}.csv")

    completed = load_completed_keys(output_csv)
    total_runs = len(graphs) * len(configs)
    current_run = 0

    print("=" * 80)
    print(f"BENCHMARK: {solver_key}")
    print(f"Graphs: {len(graphs)} | Configs: {len(configs)} | Total instances: {total_runs}")
    print(f"Time limit: {time_limit}s | Output: {output_csv}")
    if completed:
        print(f"Resuming: {len(completed)} instances already completed.")
    print("=" * 80)

    for g_idx, graph in enumerate(graphs, 1):
        for sym, refl in configs:
            current_run += 1
            enc_key = encoding or extra_kwargs.get("encoding", "")
            if (graph.name, sym, refl, enc_key) in completed:
                continue

            # Threaded live ticker using carriage return \r
            done_event = threading.Event()
            start_time = time.perf_counter()

            def ticker():
                while not done_event.is_set():
                    elapsed = time.perf_counter() - start_time
                    v_count = graph.get_vertex_count()
                    e_count = graph.get_edge_count()
                    line = (
                        f"\r[{current_run}/{total_runs}] {graph.name} "
                        f"(V={v_count}, E={e_count}) | sym={sym}, refl={refl} | "
                        f"Running... {elapsed:5.1f}s / {time_limit:.0f}s"
                    )
                    sys.stdout.write(line.ljust(80))
                    sys.stdout.flush()
                    done_event.wait(0.3)

            ticker_thread = threading.Thread(target=ticker, daemon=True)
            ticker_thread.start()

            try:
                result = solver_fn(
                    graph,
                    break_symmetry=sym,
                    break_reflection=refl,
                    time_limit=time_limit,
                    **extra_kwargs,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                result = BenchmarkResult(
                    solver_name=solver_key,
                    graph_name=graph.name,
                    vertex_count=graph.get_vertex_count(),
                    edge_count=graph.get_edge_count(),
                    symmetry_breaking=sym,
                    reflection_breaking=refl,
                    encoding=enc_key,
                    build_time=0.0,
                    solve_time=round(elapsed, 6),
                    total_time=round(elapsed, 6),
                    status="ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                )
            finally:
                done_event.set()
                ticker_thread.join()

            append_result(output_csv, result)
            completed.add((graph.name, sym, refl, enc_key))

            # Final print on completion of this instance
            v_count = graph.get_vertex_count()
            e_count = graph.get_edge_count()
            b_str = f"B={result.bandwidth}" if result.bandwidth is not None else "B=?"
            stat = result.status or "DONE"
            final_line = (
                f"\r[{current_run}/{total_runs}] {graph.name} "
                f"(V={v_count}, E={e_count}) | sym={sym}, refl={refl} | "
                f"{stat:8s} | {b_str:5s} | total={result.total_time:6.2f}s"
            )
            print(final_line.ljust(80))
            sys.stdout.flush()

    print("\nBenchmark completed. Results saved to:", output_csv)
