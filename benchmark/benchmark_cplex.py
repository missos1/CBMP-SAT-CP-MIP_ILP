from classes.benchmarkresult import BenchmarkResult
from classes.graph import Graph
from docplex.cp.model import CpoModel
from docplex.mp.model import Model
import time

def benchmap_cplex_cp(graph: Graph, break_symmetry: bool = False,
                      break_reflection: bool = False,
                      time_limit: float = 60.0) -> BenchmarkResult:
    """Benchmark Cyclic Bandwidth using CPLEX CP Optimizer (Constraint Programming)."""
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or len(edges) == 0:
        return BenchmarkResult(
            solver_name="cplex_cp",
            graph_name=graph.name,
            vertex_count=n,
            edge_count=len(edges),
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            encoding=None,
            build_time=0.0,
            solve_time=0.0,
            total_time=0.0,
            bandwidth=0,
            status="Optimal"
        )

    t_build_start = time.perf_counter()

    mdl = CpoModel(name=f"CBMP_CP_{graph.name}")

    # Variables: integer label in [1..n] for each vertex
    x = {v: mdl.integer_var(1, n, name=f"x_{v}") for v in range(1, n + 1)}

    # AllDifferent: each vertex gets a distinct label
    mdl.add(mdl.all_diff(list(x.values())))

    # Objective variable: Bandwidth B in [1..floor(n/2)]
    B = mdl.integer_var(1, n // 2, name="bandwidth")

    # Symmetry breaking (rotation): pin vertex 1 to label 1
    if break_symmetry and n > 0:
        mdl.add(x[1] == 1)

    # Reflection breaking: target vertex cannot be in second half of the cycle
    if break_reflection and break_symmetry and n > 2:
        neighbors = graph.get_neighbors(1)
        target_v = neighbors[0] if neighbors else 2
        mdl.add(x[target_v] <= (n // 2) + 1)

    # Distance constraint for each edge: min(|xu - xv|, n - |xu - xv|) <= B
    for u, v in edges:
        diff = mdl.abs(x[u] - x[v])
        cyclic_d = mdl.min(diff, n - diff)
        mdl.add(cyclic_d <= B)

    mdl.add(mdl.minimize(B))

    build_time = time.perf_counter() - t_build_start

    # Solve
    t_solve_start = time.perf_counter()
    msol = mdl.solve(TimeLimit=time_limit, LogVerbosity="Quiet")
    py_solve_time = time.perf_counter() - t_solve_start

    # Use CPLEX CP internal engine solve time if available
    if msol and hasattr(msol, "get_solve_time") and msol.get_solve_time() is not None:
        solve_time = float(msol.get_solve_time())
    else:
        solve_time = py_solve_time
    total_time = build_time + solve_time

    bandwidth = None
    status = "TIMEOUT"
    if msol:
        status = str(msol.get_solve_status())
        if msol.is_solution():
            bandwidth = int(round(msol.get_value(B)))

    return BenchmarkResult(
        solver_name="cplex_cp",
        graph_name=graph.name,
        vertex_count=n,
        edge_count=len(edges),
        symmetry_breaking=break_symmetry,
        reflection_breaking=break_reflection,
        encoding=None,
        build_time=round(build_time, 4),
        solve_time=round(solve_time, 4),
        total_time=round(total_time, 4),
        bandwidth=bandwidth,
        status=status
    )
    
def benchmap_cplex_mip(graph: Graph, break_symmetry: bool = False,
                       break_reflection: bool = False,
                       time_limit: float = 60.0) -> BenchmarkResult:
    """Benchmark Cyclic Bandwidth using CPLEX MIP/ILP (Assignment Formulation)."""
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or len(edges) == 0:
        return BenchmarkResult(
            solver_name="cplex_mip",
            graph_name=graph.name,
            vertex_count=n,
            edge_count=len(edges),
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            encoding=None,
            build_time=0.0,
            solve_time=0.0,
            total_time=0.0,
            bandwidth=0,
            status="OPTIMAL"
        )

    t_build_start = time.perf_counter()

    mdl = Model(name=f"CBMP_{graph.name}")
    mdl.parameters.timelimit = time_limit

    # Objective: Bandwidth B in [1, floor(n/2)]
    max_k = n // 2
    B = mdl.integer_var(lb=1, ub=max_k, name="bandwidth")

    # Binary assignment variables x[v, l] = 1 iff vertex v gets label l
    x = {(v, l): mdl.binary_var(name=f"x_{v}_{l}")
         for v in range(1, n + 1) for l in range(1, n + 1)}

    # Exactly one label per vertex
    for v in range(1, n + 1):
        mdl.add_constraint(mdl.sum(x[v, l] for l in range(1, n + 1)) == 1)

    # Exactly one vertex per label (permutation)
    for l in range(1, n + 1):
        mdl.add_constraint(mdl.sum(x[v, l] for v in range(1, n + 1)) == 1)

    # Symmetry breaking (rotation): pin vertex 1 to label 1
    if break_symmetry and n > 0:
        mdl.add_constraint(x[1, 1] == 1)

    # Reflection breaking: target vertex cannot take labels in second half
    if break_reflection and break_symmetry and n > 2:
        neighbors = graph.get_neighbors(1)
        target_v = neighbors[0] if neighbors else 2
        for l in range((n // 2) + 2, n + 1):
            mdl.add_constraint(x[target_v, l] == 0)

    # Cyclic distance constraint for each edge (u, v):
    # If u gets l1 and v gets l2, then B >= dist(l1, l2).
    # Linearized: B >= dist(l1, l2) * (x[u, l1] + x[v, l2] - 1)
    for u, v in edges:
        for l1 in range(1, n + 1):
            for l2 in range(1, n + 1):
                diff = abs(l1 - l2)
                d = min(diff, n - diff)
                if d > 1:  # d <= 1 is trivially <= B
                    mdl.add_constraint(B >= d * (x[u, l1] + x[v, l2] - 1))

    mdl.minimize(B)

    build_time = time.perf_counter() - t_build_start

    # Solve
    t_solve_start = time.perf_counter()
    sol = mdl.solve(log_output=False)
    py_solve_time = time.perf_counter() - t_solve_start

    # Use CPLEX internal engine time if available
    if mdl.solve_details and mdl.solve_details.time is not None:
        solve_time = float(mdl.solve_details.time)
    else:
        solve_time = py_solve_time
    total_time = build_time + solve_time

    bandwidth = None
    status = "TIMEOUT"
    if sol:
        bandwidth = int(round(sol.get_value(B)))
        status = str(sol.solve_status.name)
    elif mdl.solve_details:
        status = str(mdl.solve_details.status)

    return BenchmarkResult(
        solver_name="cplex_mip",
        graph_name=graph.name,
        vertex_count=n,
        edge_count=len(edges),
        symmetry_breaking=break_symmetry,
        reflection_breaking=break_reflection,
        encoding=None,
        build_time=round(build_time, 4),
        solve_time=round(solve_time, 4),
        total_time=round(total_time, 4),
        bandwidth=bandwidth,
        status=status
    )
    
