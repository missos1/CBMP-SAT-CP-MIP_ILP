"""Validation of CBMP labelings and cyclic bandwidth evaluation."""

from cbmp_cnf import cyclic_dist


def is_bijection(graph, labels) -> bool:
    vertices = set(graph.adjacency_list)
    if set(labels) != vertices:
        return False
    n = len(vertices)
    return sorted(labels.values()) == list(range(1, n + 1))


def cyclic_bandwidth(graph, labels) -> int:
    """Return ``CB(G, labels)``; raise ``ValueError`` on an invalid labeling."""
    vertices = set(graph.adjacency_list)
    if set(labels) != vertices:
        raise ValueError("labels must define exactly the graph vertices")
    n = len(vertices)
    if sorted(labels.values()) != list(range(1, n + 1)):
        raise ValueError("labels must be a bijection onto 1..n")
    return max(
        (cyclic_dist(labels[u], labels[v], n) for u, v in graph.get_edges()),
        default=0,
    )


def verify_labeling(graph, labels):
    """Return ``(ok, bandwidth, message)`` for a candidate labeling."""
    try:
        return True, cyclic_bandwidth(graph, labels), ""
    except ValueError as exc:
        return False, None, str(exc)
