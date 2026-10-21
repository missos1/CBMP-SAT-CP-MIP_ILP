class Graph:
    def __init__(self):
        self.adjacency_list = {}

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

    def __str__(self):
        return str(self.adjacency_list)