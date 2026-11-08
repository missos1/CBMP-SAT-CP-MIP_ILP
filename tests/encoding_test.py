import itertools

import pytest
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

from cbmp_cnf import build_cnf, cyclic_dist, decode_model, pairwise_amo
from classes.graph import Graph


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


def star_graph(n):
    return make_graph(f"star{n}", n, [(1, i) for i in range(2, n + 1)])


def complete_graph(n):
    return make_graph(f"k{n}", n, itertools.combinations(range(1, n + 1), 2))


def brute_min_bandwidth(graph):
    vertices = list(graph.adjacency_list)
    n = len(vertices)
    edges = graph.get_edges()
    best = None
    for perm in itertools.permutations(range(1, n + 1)):
        labels = dict(zip(vertices, perm))
        cost = max(
            (cyclic_dist(labels[u], labels[v], n) for u, v in edges), default=0
        )
        if best is None or cost < best:
            best = cost
    return best


def solve(cnf):
    with Solver(name="minisat22", bootstrap_with=cnf.clauses) as solver:
        ok = solver.solve()
        model = solver.get_model() if ok else None
    return ok, model


def model_bandwidth(graph, labels):
    vertices = list(graph.adjacency_list)
    n = len(vertices)
    return max(
        (cyclic_dist(labels[u], labels[v], n) for u, v in graph.get_edges()),
        default=0,
    )


def test_pairwise_amo_matches_reference():
    lits = [1, 2, 3, 4, 5, 6, 7]
    custom = CNF()
    pairwise_amo(lits, custom)
    reference = CardEnc.atmost(lits=lits, encoding=EncType.pairwise)
    assert {frozenset(c) for c in custom.clauses} == {
        frozenset(c) for c in reference.clauses
    }


GRAPHS = [
    path_graph(4),
    path_graph(6),
    cycle_graph(5),
    cycle_graph(6),
    star_graph(5),
    complete_graph(4),
]


@pytest.mark.parametrize("graph", GRAPHS, ids=lambda g: g.name)
@pytest.mark.parametrize("encoding", ["pairwise", "order"])
def test_decision_matches_bruteforce(graph, encoding):
    optimum = brute_min_bandwidth(graph)
    n = graph.get_vertex_count()
    for bound in range(0, n // 2 + 1):
        cnf = build_cnf(graph, bound, encoding=encoding)
        sat, _ = solve(cnf)
        assert sat == (optimum <= bound), (
            graph.name, encoding, bound, optimum, sat
        )


@pytest.mark.parametrize("graph", GRAPHS, ids=lambda g: g.name)
def test_encodings_agree(graph):
    n = graph.get_vertex_count()
    for bound in range(0, n // 2 + 1):
        sat_pairwise, _ = solve(build_cnf(graph, bound, encoding="pairwise"))
        sat_order, _ = solve(build_cnf(graph, bound, encoding="order"))
        assert sat_pairwise == sat_order, (graph.name, bound)


@pytest.mark.parametrize("graph", GRAPHS, ids=lambda g: g.name)
@pytest.mark.parametrize("encoding", ["pairwise", "order"])
def test_decode_reaches_optimum(graph, encoding):
    optimum = brute_min_bandwidth(graph)
    sat, model = solve(build_cnf(graph, optimum, encoding=encoding))
    assert sat
    labels = decode_model(model, graph, encoding=encoding)
    assert sorted(labels.values()) == list(
        range(1, graph.get_vertex_count() + 1)
    )
    assert model_bandwidth(graph, labels) == optimum


@pytest.mark.parametrize("graph", GRAPHS, ids=lambda g: g.name)
@pytest.mark.parametrize("encoding", ["pairwise", "order"])
def test_symmetry_breaking_preserves_optimum(graph, encoding):
    optimum = brute_min_bandwidth(graph)
    for bound in range(0, graph.get_vertex_count() // 2 + 1):
        base, _ = solve(build_cnf(graph, bound, encoding=encoding))
        broken, _ = solve(
            build_cnf(
                graph, bound, encoding=encoding,
                break_symmetry=True, break_reflection=True,
            )
        )
        assert base == broken, (graph.name, encoding, bound)


def test_pairwise_amo_variants():
    graph = cycle_graph(6)
    for amo in ["pairwise", "seqcounter", "totalizer"]:
        cnf = build_cnf(graph, 1, encoding="pairwise", amo_encoding=amo)
        sat, _ = solve(cnf)
        assert sat


def test_order_cardinality_variants():
    graph = cycle_graph(6)
    for card in ["seqcounter", "totalizer", "cardnetwrk"]:
        cnf = build_cnf(graph, 1, encoding="order", card_encoding=card)
        sat, _ = solve(cnf)
        assert sat
