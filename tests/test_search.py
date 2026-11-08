import itertools
from pathlib import Path

import pytest

from cbmp_search import (
    SAT,
    UNSAT,
    find_min_bandwidth,
    find_min_bandwidth_incremental,
    lower_bound,
    solve_decision,
    upper_bound,
)
from classes.graph import Graph
from classes.verify import cyclic_bandwidth, is_bijection, verify_labeling

ROOT = Path(__file__).resolve().parents[1]
SOLVER = "minisat22"


def make_graph(name, n, edges):
    graph = Graph(name, n)
    for u, v in edges:
        graph.add_edge(u, v)
    return graph


def path_graph(n):
    return make_graph(f"path{n}", n, [(i, i + 1) for i in range(1, n)])


def cycle_graph(n):
    edges = [(i, i + 1) for i in range(1, n)] + [(n, 1)]
    return make_graph(f"cycle{n}", n, edges)


def mesh2d(rows, cols):
    graph = Graph(f"mesh2D{rows}x{cols}", rows * cols)

    def vid(r, c):
        return r * cols + c + 1

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                graph.add_edge(vid(r, c), vid(r, c + 1))
            if r + 1 < rows:
                graph.add_edge(vid(r, c), vid(r + 1, c))
    return graph


def brute_min(graph):
    vertices = list(graph.adjacency_list)
    n = len(vertices)
    best = None
    for perm in itertools.permutations(range(1, n + 1)):
        cost = cyclic_bandwidth(graph, dict(zip(vertices, perm)))
        if best is None or cost < best:
            best = cost
    return best


def test_verify_labeling():
    graph = path_graph(4)
    ok, cb, message = verify_labeling(graph, {1: 1, 2: 2, 3: 3, 4: 4})
    assert ok and cb == 1 and message == ""
    assert is_bijection(graph, {1: 4, 2: 3, 3: 2, 4: 1})

    ok, cb, message = verify_labeling(graph, {1: 1, 2: 1, 3: 3, 4: 4})
    assert not ok and cb is None and "bijection" in message

    ok, cb, message = verify_labeling(graph, {1: 1, 2: 2, 3: 3})
    assert not ok and cb is None


def test_bounds():
    graph = path_graph(10)
    assert lower_bound(graph) == 1
    assert upper_bound(graph) == 5

    graph = mesh2d(5, 4)
    assert lower_bound(graph) == 2
    assert upper_bound(graph) == 10


def test_solve_decision_status():
    graph = cycle_graph(6)
    assert solve_decision(graph, 1, solver_name=SOLVER).status == SAT
    assert solve_decision(graph, 0, solver_name=SOLVER).status == UNSAT


@pytest.mark.parametrize("graph", [path_graph(8), cycle_graph(8)],
                         ids=lambda g: g.name)
@pytest.mark.parametrize("encoding", ["pairwise", "order"])
def test_path_and_cycle_optimum(graph, encoding):
    result = find_min_bandwidth(graph, encoding=encoding, solver_name=SOLVER)
    assert result.status == "OPTIMAL"
    assert result.optimal
    assert result.bandwidth == 1
    assert cyclic_bandwidth(graph, result.labels) == 1


@pytest.mark.parametrize("encoding", ["pairwise", "order"])
def test_mesh2d_optimum(encoding):
    graph = mesh2d(5, 4)
    result = find_min_bandwidth(graph, encoding=encoding, solver_name=SOLVER)
    assert result.optimal and result.bandwidth == 4


@pytest.mark.parametrize("encoding", ["pairwise", "order"])
def test_harwell_boeing_known_optima(encoding):
    cases = {
        "counterx/jgl009.mtx.gz": 4,
        "counterx/rgg010.mtx.gz": 5,
        "counterx/jgl011.mtx.gz": 5,
    }
    from mtx_parser import parse_file

    for relative, expected in cases.items():
        name, n, edges = parse_file(ROOT / "dataset" / relative)
        graph = Graph.from_parsed((name, n, edges))
        result = find_min_bandwidth(graph, encoding=encoding, solver_name=SOLVER)
        assert result.optimal, (name, result)
        assert result.bandwidth == expected, (name, result)


@pytest.mark.parametrize("graph", [path_graph(6), cycle_graph(6), mesh2d(4, 4)],
                         ids=lambda g: g.name)
def test_incremental_matches_binary_search(graph):
    exact = find_min_bandwidth(graph, encoding="pairwise", solver_name=SOLVER)
    incremental = find_min_bandwidth_incremental(graph, solver_name=SOLVER)
    assert incremental.optimal
    assert incremental.bandwidth == exact.bandwidth
    assert cyclic_bandwidth(graph, incremental.labels) == incremental.bandwidth


def test_trivial_graph_without_edges():
    graph = make_graph("isolated", 3, [])
    result = find_min_bandwidth(graph, solver_name=SOLVER)
    assert result.status == "TRIVIAL"
    assert result.optimal and result.bandwidth == 0


def test_matches_bruteforce_small():
    for graph in [path_graph(5), cycle_graph(5), mesh2d(2, 3)]:
        expected = brute_min(graph)
        for encoding in ["pairwise", "order"]:
            result = find_min_bandwidth(graph, encoding=encoding, solver_name=SOLVER)
            assert result.bandwidth == expected, (graph.name, encoding)

def test_subprocess_time_limited_decision():
    graph = cycle_graph(5)
    result = solve_decision(graph, 1, solver_name=SOLVER, time_limit=30)
    assert result.status == SAT
    assert result.bandwidth == 1
