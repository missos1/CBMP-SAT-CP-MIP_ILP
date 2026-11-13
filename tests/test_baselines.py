import itertools
from functools import lru_cache
from pathlib import Path

import pytest

from classes.graph import Graph
from classes.verify import cyclic_bandwidth


ROOT = Path(__file__).resolve().parents[1]
TIME_LIMIT = 60.0


@lru_cache(maxsize=1)
def gurobi_available() -> bool:
    try:
        import gurobipy as gp
        model = gp.Model("probe")
        model.Params.OutputFlag = 0
        model.dispose()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def cplex_available() -> bool:
    try:
        from docplex.mp.model import Model
        model = Model()
        value = model.integer_var(lb=1)
        model.minimize(value)
        model.solve(log_output=False)
        model.end()
        return True
    except Exception:
        return False


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


def brute_optimum(graph):
    vertices = list(graph.adjacency_list)
    best = None
    for perm in itertools.permutations(range(1, len(vertices) + 1)):
        cost = cyclic_bandwidth(graph, dict(zip(vertices, perm)))
        if best is None or cost < best:
            best = cost
    return best


def default_baselines():
    from benchmark.benchmark_cplex import benchmap_cplex_cp, benchmap_cplex_mip
    from benchmark.benchmark_gurobi import benchmap_gurobi_mip
    from benchmark.benchmark_ortools import benchmap_cpsat

    return [
        ("gurobi_mip", benchmap_gurobi_mip, gurobi_available),
        ("cplex_mip", benchmap_cplex_mip, cplex_available),
        ("cplex_cp", benchmap_cplex_cp, cplex_available),
        ("cpsat", benchmap_cpsat, lambda: True),
    ]


BASELINES = default_baselines()
BASELINE_IDS = [name for name, _, _ in BASELINES]


def run_baseline(index, graph, **kwargs):
    name, function, available = BASELINES[index]
    if not available():
        pytest.skip(f"{name} not available")
    return function(graph, time_limit=TIME_LIMIT, **kwargs)


@pytest.mark.parametrize("index", range(len(BASELINES)), ids=BASELINE_IDS)
@pytest.mark.parametrize("graph", [path_graph(5), cycle_graph(5),
                                   mesh2d(2, 3)],
                         ids=lambda g: g.name)
def test_baselines_match_bruteforce(index, graph):
    expected = brute_optimum(graph)
    result = run_baseline(index, graph)
    assert result.status in ("OPTIMAL", "FEASIBLE"), result
    assert result.verified, result.message
    assert result.bandwidth == expected
    assert result.lower_bound <= expected <= result.upper_bound
    if result.status == "OPTIMAL":
        assert result.optimal


@pytest.mark.parametrize("index", range(len(BASELINES)), ids=BASELINE_IDS)
def test_baselines_known_mesh_optima(index):
    for graph, expected in [(mesh2d(3, 3), 3), (mesh2d(4, 4), 4)]:
        result = run_baseline(index, graph)
        assert result.status in ("OPTIMAL", "FEASIBLE"), (graph.name, result)
        assert result.verified, (graph.name, result.message)
        assert result.bandwidth == expected, (graph.name, result)


@pytest.mark.parametrize("index", range(len(BASELINES)), ids=BASELINE_IDS)
def test_baselines_symmetry_breaking(index):
    graph = mesh2d(4, 4)
    result = run_baseline(index, graph, break_symmetry=True,
                          break_reflection=True)
    assert result.status in ("OPTIMAL", "FEASIBLE"), result
    assert result.verified
    assert result.bandwidth == 4


@pytest.mark.parametrize("index", range(len(BASELINES)), ids=BASELINE_IDS)
def test_baselines_harwell_boeing(index):
    from benchmark.mtx_parser import parse_file

    cases = {
        "counterx/jgl009.mtx.gz": 4,
        "counterx/rgg010.mtx.gz": 5,
    }
    for relative, expected in cases.items():
        name, n, edges = parse_file(ROOT / "dataset" / relative)
        graph = Graph.from_parsed((name, n, edges))
        result = run_baseline(index, graph)
        assert result.verified, (name, result.message)
        assert result.bandwidth == expected, (name, result)
