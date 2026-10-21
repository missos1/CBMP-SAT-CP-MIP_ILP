from pysat.formula import CNF
from graph import Graph


def cyclic_dist(l1: int, l2: int, n: int) -> int:
    return min(abs(l1 - l2), n - abs(l1 - l2))

def var_encode(label: int, vertex: int, vertex_count: int) -> int:
    return (label - 1) * vertex_count + vertex

def cyclic_bandwidth_cnf(graph: Graph, min_bandwidth: int, encoding: str, 
                         use_additional_alo: bool=False,
                         break_symmetry: bool=False) -> CNF:
    enc_amo = {
        "pairwise": pairwise_amo,
        "order": order_amo
    }
    
    vertex_cnt = graph.get_vertex_count()
    labels = list(range(1, vertex_cnt + 1))
    
    #Vertex might not be int so we map it to int
    vertex_indices = {key : i for i, key in enumerate(graph.adjacency_list, start=1)}

    cnf = CNF()
    
    # Each vertex must have exactly one label
    for vertex in graph.adjacency_list.keys():
        group = [var_encode(label, vertex_indices[vertex], vertex_cnt) for label in labels]
        cnf.append(group)
        enc_amo[encoding](group, cnf)
    
    # AMO: Each label must belongs to at most one vertex
    for label in labels:
        group = [var_encode(label, vertex_indices[vertex], vertex_cnt) for vertex in graph.adjacency_list.keys()]
        # Additional ALO may help with propagation but not needed due to pigeonhole principle
        if use_additional_alo: 
            cnf.append(group)
        enc_amo[encoding](group, cnf)
        
    
    for u, neighbors in graph.adjacency_list.items():
        u_idx = vertex_indices[u]
        for v in neighbors:
            v_idx = vertex_indices[v]
            if u_idx < v_idx:  # process each undirected edge once
                for l1 in labels:
                    for l2 in labels:
                        if cyclic_dist(l1, l2, vertex_cnt) > min_bandwidth:
                            cnf.append([
                                -var_encode(l1, u_idx, vertex_cnt),
                                -var_encode(l2, v_idx, vertex_cnt)
                            ])
    if break_symmetry and vertex_cnt > 0:
        cnf.append(var_encode(1, 1, vertex_cnt))
    return cnf
    
    
def pairwise_amo(group: list, cnf : CNF) -> None:
    pass

def order_amo(group: list, cnf : CNF) -> None:
    pass
    