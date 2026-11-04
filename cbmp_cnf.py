from pysat.formula import CNF
from classes.graph import Graph


def cyclic_dist(l1: int, l2: int, n: int) -> int:
    return min(abs(l1 - l2), n - abs(l1 - l2))

def var_encode(label: int, vertex: int, vertex_count: int) -> int:
    return (label - 1) * vertex_count + vertex

def base_cnf(graph: Graph, encoding: str, break_symmetry: bool=False, break_reflection: bool=False) -> CNF:
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
    
    # Each label must belongs exactly one vertex
    for label in labels:
        group = [var_encode(label, vertex_indices[vertex], vertex_cnt) for vertex in graph.adjacency_list.keys()]
        cnf.append(group)
        enc_amo[encoding](group, cnf)
    
    if break_symmetry and vertex_cnt > 0:
        cnf.append(var_encode(1, 1, vertex_cnt))
        
        if break_reflection and vertex_cnt > 1:
            neighbors = graph.get_neighbors(1)
            target_v = neighbors[0] if neighbors else 2
            
            # Forbid target_v from having labels in the second half of the circle
            max_allowed_label = (vertex_cnt // 2) + 1
            for label in range(max_allowed_label + 1, vertex_cnt + 1):
                cnf.append([-var_encode(label, target_v, vertex_cnt)])
    return cnf
    
def pairwise_amo(group: list, cnf : CNF) -> None:
    pass

def order_amo(group: list, cnf : CNF) -> None:
    pass
    