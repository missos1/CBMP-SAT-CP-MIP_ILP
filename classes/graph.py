class Graph:
    def __init__(self, name: str = "", vertex_count: int = 0):
        """Undirected graph. If vertex_count is given, vertices 1..vertex_count
        are created up front (so isolated vertices are kept, in order)."""
        self.name = name
        self.adjacency_list = {}
        for v in range(1, vertex_count + 1):
            self.adjacency_list[v] = []

    @classmethod
    def from_parsed(cls, parsed) -> "Graph":
        """Build from mtx_parser output: (dataset_name, vertex_count, edges)."""
        name, vertex_count, edges = parsed
        graph = cls(name, vertex_count)
        for u, v in edges:
            graph.add_edge(u, v)
        return graph

    def add_edge(self, u, v):
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)  # For undirected graph
        
    def get_edges(self):
        edges = []
        for u, neighbors in self.adjacency_list.items():
            for v in neighbors:
                if u < v:  
                    edges.append((u, v))
        return edges
    
    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def get_neighbors(self, node):
        return self.adjacency_list.get(node, [])
    
    def get_vertex_count(self):
        return len(self.adjacency_list)
    
    def get_edge_count(self):
        return sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2  # Each edge is counted twice

    def get_density(self) -> float:
        n = self.get_vertex_count()
        if n <= 1:
            return 0.0
        return (2 * self.get_edge_count()) / (n * (n - 1))

    def __str__(self):
        return str(self.adjacency_list)

    def __repr__(self):
        return (f"Graph(name={self.name!r}, vertices={self.get_vertex_count()}, "
                f"edges={self.get_edge_count()}, density={self.get_density():.3f})")


def estimated_clauses(vertex_count: int, edge_count: int) -> int:
    """Rough size of the forbidden-distance clauses: about |E| * n^2."""
    return edge_count * vertex_count * vertex_count


def graph_density(vertex_count: int, edge_count: int) -> float:
    """Edge density: 2 * |E| / (n * (n - 1))."""
    if vertex_count <= 1:
        return 0.0
    return (2 * edge_count) / (vertex_count * (vertex_count - 1))


def load_graphs(root: str = "dataset", skipped: list = None,
                max_clauses: int = 50_000_000, max_edges: int = None,
                max_density: float = 0.8,
                sort_by_edges: bool = True) -> list:
    """Load every usable matrix under `root` as a Graph (see
    mtx_parser.parse_all for what is skipped; `skipped` collects the reasons).

    Graphs are dropped (reason recorded in `skipped`) if:
      * estimated_clauses(n, |E|) > max_clauses
      * |E| > max_edges
      * graph_density(n, |E|) >= max_density (near-cliques, trivial cyclic bandwidth)
    Pass None to disable any limit. The rest are sorted by edge count.
    """
    from benchmark.mtx_parser import parse_all

    def drop(name, reason):
        if skipped is not None:
            skipped.append((name, reason))

    graphs = []
    for parsed in parse_all(root, skipped=skipped):
        name, n, edges = parsed
        m = len(edges)
        est = estimated_clauses(n, m)
        dens = graph_density(n, m)
        if max_density is not None and dens >= max_density:
            drop(name, f"density {dens:.3f} >= {max_density} (near-clique)")
        elif max_clauses is not None and est > max_clauses:
            drop(name, f"~{est:.2e} clauses > {max_clauses:.2e}")
        elif max_edges is not None and m > max_edges:
            drop(name, f"{m} edges > {max_edges}")
        else:
            graphs.append(Graph.from_parsed(parsed))
    if sort_by_edges:
        graphs.sort(key=lambda g: (g.get_edge_count(), g.get_vertex_count()))
    return graphs
