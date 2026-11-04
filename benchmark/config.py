from classes.graph import load_graphs
# Benchmark configuration file
# Since break reflection is only applicable when break symmetry is enabled, we will only benchmark the following configurations:
CONFIG_LIST = [(False, False), (True, False), (True, True)]
TIME_LIMIT = 500.0  # seconds
DENSITY_LIMIT = 0.8  # maximum density of graphs to benchmark
MAX_CLAUSES = 50_000_000  # maximum number of clauses to benchmark
GRAPH_LIST = load_graphs(
    root="dataset", 
    skipped=[], 
    max_clauses=50_000_000, 
    max_edges=None, 
    max_density=DENSITY_LIMIT, 
    sort_by_edges=True)