"""SAT encodings for the Cyclic Bandwidth Minimization Problem (CBMP).

Two encodings are provided:

* ``pairwise``: one-hot assignment ``x[v, l]`` (vertex ``v`` gets label ``l``)
  with exactly-one constraints, plus one forbidden-pair clause for every edge
  and every pair of labels whose cyclic distance exceeds the bound ``B``.
* ``order``: order literals ``o[v, k] = [label(v) >= k]`` with monotonicity,
  all-different enforced through per-threshold cardinality constraints, and
  the cyclic distance bound encoded with three mutually exclusive regions.

Labels are ``1..n`` and vertices are indexed ``1..n`` by their position in
``graph.adjacency_list``.
"""

from pysat.card import CardEnc, EncType
from pysat.formula import CNF

from classes.graph import Graph


ENCODINGS = ("pairwise", "order")

_ENCODING_NAMES = {
    "pairwise": EncType.pairwise,
    "seqcounter": EncType.seqcounter,
    "sequential": EncType.seqcounter,
    "sortnetwrk": EncType.sortnetwrk,
    "cardnetwrk": EncType.cardnetwrk,
    "totalizer": EncType.totalizer,
    "kmtotalizer": EncType.kmtotalizer,
    "mtotalizer": EncType.mtotalizer,
    "ladder": EncType.ladder,
    "bitwise": EncType.bitwise,
    "native": EncType.native,
}


def cyclic_dist(a: int, b: int, n: int) -> int:
    """Cyclic distance ``min(|a-b|, n-|a-b|)`` on a cycle of length ``n``."""
    d = abs(a - b)
    return min(d, n - d)


def var_encode(label: int, vertex: int, vertex_count: int) -> int:
    """One-hot variable id for ``(label, vertex)`` with ``label, vertex`` 1-based."""
    return (label - 1) * vertex_count + vertex


def order_var(vertex: int, k: int, vertex_count: int) -> int:
    """Order variable id for ``[label(vertex) >= k]`` (``vertex`` 1-based)."""
    return (vertex - 1) * (vertex_count - 1) + (k - 1)


def _encoding_value(encoding) -> int:
    if isinstance(encoding, int):
        return encoding
    try:
        return _ENCODING_NAMES[str(encoding).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown encoding: {encoding!r}") from exc


def _max_var(cnf: CNF, lits) -> int:
    top = cnf.nv
    for lit in lits:
        top = max(top, abs(lit))
    return top


def pairwise_amo(group, cnf: CNF) -> None:
    """At-most-one over ``group`` using all pairwise negative clauses."""
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            cnf.append([-group[i], -group[j]])


def add_amo(group, cnf: CNF, encoding="pairwise") -> None:
    """Add an at-most-one constraint over ``group`` with a chosen encoding."""
    if len(group) <= 1:
        return
    enc = _encoding_value(encoding)
    if enc == EncType.pairwise:
        pairwise_amo(group, cnf)
        return
    result = CardEnc.atmost(
        lits=list(group), bound=1, top_id=_max_var(cnf, group), encoding=enc
    )
    for clause in result.clauses:
        cnf.append(clause)


def _add_exactly(cnf: CNF, lits, bound: int, encoding) -> None:
    result = CardEnc.equals(
        lits=list(lits), bound=bound,
        top_id=_max_var(cnf, lits), encoding=_encoding_value(encoding),
    )
    for clause in result.clauses:
        cnf.append(clause)


def _vertex_order(graph: Graph):
    vertices = list(graph.adjacency_list.keys())
    index = {v: i for i, v in enumerate(vertices, 1)}
    return vertices, index, len(vertices)


def _add_symmetry_pairwise(cnf, graph, index, n,
                           break_symmetry, break_reflection):
    if not break_symmetry or n == 0:
        return
    keys = list(graph.adjacency_list.keys())
    first = keys[0]
    cnf.append([var_encode(1, index[first], n)])
    if break_reflection and n > 2:
        neighbours = graph.get_neighbors(first)
        target = neighbours[0] if neighbours else keys[1]
        limit = n // 2 + 1
        for label in range(limit + 1, n + 1):
            cnf.append([-var_encode(label, index[target], n)])


def _add_symmetry_order(cnf, graph, index, order, n,
                        break_symmetry, break_reflection):
    if not break_symmetry or n == 0:
        return
    keys = list(graph.adjacency_list.keys())
    first = keys[0]
    for k in range(2, n + 1):
        cnf.append([-order[(index[first], k)]])
    if break_reflection and n > 2:
        neighbours = graph.get_neighbors(first)
        target = neighbours[0] if neighbours else keys[1]
        limit = n // 2 + 1
        for k in range(limit + 1, n + 1):
            cnf.append([-order[(index[target], k)]])


def _forbidden_labels(label: int, bound: int, n: int):
    """Labels ``q`` with ``cyclic_dist(label, q, n) > bound`` (the forbidden arc)."""
    count = n - (2 * bound + 1)
    if count <= 0:
        return
    start = ((label - 1 + bound + 1) % n) + 1
    for t in range(count):
        yield ((start - 1 + t) % n) + 1


def labels_at_distance(label: int, distance: int, n: int):
    """Yield the one or two labels at exactly ``distance`` from ``label``."""
    if distance <= 0 or distance > n // 2:
        return
    forward = ((label - 1 + distance) % n) + 1
    backward = ((label - 1 - distance) % n) + 1
    yield forward
    if backward != forward:
        yield backward


def build_pairwise_cnf(graph: Graph, bound: int, break_symmetry: bool = False,
                       break_reflection: bool = False,
                       amo_encoding="pairwise") -> CNF:
    """One-hot (pairwise forbidden-pair) encoding of ``CB(G) <= bound``."""
    vertices, index, n = _vertex_order(graph)
    cnf = CNF()
    if n == 0:
        return cnf

    for i in range(1, n + 1):
        group = [var_encode(label, i, n) for label in range(1, n + 1)]
        cnf.append(group)
        add_amo(group, cnf, amo_encoding)

    for label in range(1, n + 1):
        group = [var_encode(label, i, n) for i in range(1, n + 1)]
        cnf.append(group)
        add_amo(group, cnf, amo_encoding)

    for u, v in graph.get_edges():
        iu, iv = index[u], index[v]
        for label_u in range(1, n + 1):
            for label_v in _forbidden_labels(label_u, bound, n):
                cnf.append([
                    -var_encode(label_u, iu, n),
                    -var_encode(label_v, iv, n),
                ])

    _add_symmetry_pairwise(cnf, graph, index, n,
                           break_symmetry, break_reflection)
    return cnf


def _add_cyclic_constraints_order(cnf, graph, index, order, n, bound):
    c = n - bound
    top = cnf.nv

    def new_var():
        nonlocal top
        top += 1
        return top

    for u, v in graph.get_edges():
        iu, iv = index[u], index[v]
        sel_a = new_var()
        sel_b = new_var()
        sel_c = new_var()
        cnf.append([sel_a, sel_b, sel_c])

        for k in range(2, n + 1):
            j = k - bound
            if j >= 2:
                cnf.append([-sel_a, -order[(iv, k)], order[(iu, j)]])
                cnf.append([-sel_a, -order[(iu, k)], order[(iv, j)]])

            j2 = k + c
            if j2 <= n:
                cnf.append([-sel_b, -order[(iu, k)], order[(iv, j2)]])
                cnf.append([-sel_c, -order[(iv, k)], order[(iu, j2)]])
            else:
                cnf.append([-sel_b, -order[(iu, k)]])
                cnf.append([-sel_c, -order[(iv, k)]])

        j1 = 1 + c
        if j1 == 1:
            cnf.append([-sel_b])
            cnf.append([-sel_c])
        elif j1 <= n:
            cnf.append([-sel_b, order[(iv, j1)]])
            cnf.append([-sel_c, order[(iu, j1)]])
        else:
            cnf.append([-sel_b])
            cnf.append([-sel_c])

    cnf.nv = max(cnf.nv, top)


def build_order_cnf(graph: Graph, bound: int, break_symmetry: bool = False,
                    break_reflection: bool = False,
                    card_encoding="seqcounter") -> CNF:
    """Order encoding of ``CB(G) <= bound`` (true integer-variable order encoding)."""
    vertices, index, n = _vertex_order(graph)
    cnf = CNF()
    if n == 0:
        return cnf

    order = {}
    for i in range(1, n + 1):
        for k in range(2, n + 1):
            order[(i, k)] = order_var(i, k, n)

    for i in range(1, n + 1):
        for k in range(3, n + 1):
            cnf.append([-order[(i, k)], order[(i, k - 1)]])

    for k in range(2, n + 1):
        lits = [order[(i, k)] for i in range(1, n + 1)]
        _add_exactly(cnf, lits, n - k + 1, card_encoding)

    if n > 1 and bound < n // 2:
        _add_cyclic_constraints_order(cnf, graph, index, order, n, bound)

    _add_symmetry_order(cnf, graph, index, order, n,
                        break_symmetry, break_reflection)

    cnf.nv = max(cnf.nv, n * (n - 1))
    return cnf


def build_cnf(graph: Graph, bound: int, encoding: str = "pairwise",
              break_symmetry: bool = False, break_reflection: bool = False,
              amo_encoding="pairwise", card_encoding="seqcounter") -> CNF:
    """Build a CNF for the decision problem ``CB(G) <= bound``."""
    if encoding == "pairwise":
        return build_pairwise_cnf(
            graph, bound, break_symmetry, break_reflection, amo_encoding
        )
    if encoding == "order":
        return build_order_cnf(
            graph, bound, break_symmetry, break_reflection, card_encoding
        )
    raise ValueError(f"unknown encoding: {encoding!r}")


def decode_model(model, graph: Graph, encoding: str = "pairwise") -> dict:
    """Decode a PySAT model into a ``{vertex: label}`` mapping."""
    vertices, index, n = _vertex_order(graph)
    positive = {lit for lit in model if lit > 0}
    labels = {}

    if encoding == "pairwise":
        for key, i in index.items():
            for label in range(1, n + 1):
                if var_encode(label, i, n) in positive:
                    labels[key] = label
                    break
    elif encoding == "order":
        for key, i in index.items():
            value = 1
            for k in range(2, n + 1):
                if order_var(i, k, n) in positive:
                    value += 1
            labels[key] = value
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    return labels
