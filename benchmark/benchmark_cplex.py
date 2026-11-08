"""CPLEX CP and MIP/ILP baselines for CBMP."""

import math
import time

from docplex.cp.model import CpoModel
from docplex.mp.model import Model

from benchmark.common import (
    ERROR,
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    TIMEOUT,
    UNKNOWN,
    finalize_result,
    theoretical_lower_bound,
)
from classes.graph import Graph


def _cp_status(solution, time_limit: float) -> str:
    if solution is None:
        return UNKNOWN
    raw = str(solution.get_solve_status()).lower()
    if "optimal" in raw:
        return OPTIMAL
    if "feasible" in raw:
        return FEASIBLE
    if "infeasible" in raw:
        return INFEASIBLE
    if "unknown" in raw:
        elapsed = 0.0
        try:
            elapsed = float(solution.get_solve_time())
        except Exception:
            pass
        return TIMEOUT if elapsed >= time_limit - 1e-6 else UNKNOWN
    return UNKNOWN


def _mip_status(solution, details) -> str:
    if solution is not None:
        raw = str(solution.solve_status.name).lower()
        if "optimal" in raw:
            return OPTIMAL
        if "feasible" in raw:
            return FEASIBLE
        if "infeasible" in raw:
            return INFEASIBLE
        return UNKNOWN
    raw = str(details.status).lower() if details is not None else ""
    if "infeasible" in raw:
        return INFEASIBLE
    if "time" in raw or "limit" in raw:
        return TIMEOUT
    return UNKNOWN


def benchmap_cplex_cp(graph: Graph, break_symmetry: bool = False,
                      break_reflection: bool = False,
                      time_limit: float = 60.0) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="cplex_cp", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    start = time.perf_counter()

    try:
        mdl = CpoModel(name="cbmp_cplex_cp")
        x = {v: mdl.integer_var(1, n, name=f"x_{i}")
             for i, v in enumerate(vertices)}
        mdl.add(mdl.all_diff(list(x.values())))

        if break_symmetry:
            mdl.add(x[vertices[0]] == 1)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = neighbours[0] if neighbours else vertices[1]
                mdl.add(x[target] <= n // 2 + 1)

        bound = mdl.integer_var(0, n // 2, name="B")
        for u, v in edges:
            diff = mdl.abs(x[u] - x[v])
            mdl.add(mdl.min(diff, n - diff) <= bound)
        mdl.add(mdl.minimize(bound))

        build_time = time.perf_counter() - start
        solution = mdl.solve(TimeLimit=float(time_limit), LogVerbosity="Quiet")
        solve_time = time.perf_counter() - start - build_time
        if solution is not None:
            try:
                solve_time = float(solution.get_solve_time())
            except Exception:
                pass

        status = _cp_status(solution, time_limit)
        labels = None
        if solution is not None and solution.is_solution():
            labels = {
                key: int(round(solution.get_value(x[key])))
                for key in vertices
            }

        return finalize_result(
            solver_name="cplex_cp", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=status, labels=labels,
            lower_bound=theoretical_lower_bound(graph),
            build_time=build_time, solve_time=solve_time,
            message="" if solution is not None else "no solution returned",
        )

    except Exception as exc:
        return finalize_result(
            solver_name="cplex_cp", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
        )


def benchmap_cplex_mip(graph: Graph, break_symmetry: bool = False,
                       break_reflection: bool = False,
                       time_limit: float = 60.0,
                       mip_gap: float = 0.0) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="cplex_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    index = {v: i for i, v in enumerate(vertices)}
    start = time.perf_counter()

    try:
        mdl = Model(name="cbmp_cplex_mip")
        mdl.parameters.timelimit = float(time_limit)
        mdl.parameters.mip.tolerances.mipgap = float(mip_gap)

        x = {(i, l): mdl.binary_var(name=f"x_{i}_{l}")
             for i in range(n) for l in range(n)}
        for i in range(n):
            mdl.add_constraint(mdl.sum(x[i, l] for l in range(n)) == 1)
        for l in range(n):
            mdl.add_constraint(mdl.sum(x[i, l] for i in range(n)) == 1)

        if break_symmetry:
            mdl.add_constraint(x[0, 0] == 1)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = index[neighbours[0]] if neighbours else 1
                mdl.add_constraint(
                    mdl.sum(x[target, l] for l in range(n // 2 + 1, n)) == 0
                )

        bound = mdl.integer_var(lb=0, ub=n // 2, name="B")
        position = mdl.integer_var_list(n, 0, n - 1, name="p")
        for i in range(n):
            mdl.add_constraint(
                position[i] == mdl.sum(l * x[i, l] for l in range(n))
            )

        for edge_index, (u, v) in enumerate(edges):
            iu, iv = index[u], index[v]
            offset = mdl.integer_var(lb=0, ub=n - 1, name=f"d_{edge_index}")
            wrap = mdl.binary_var(name=f"w_{edge_index}")
            side = mdl.binary_var(name=f"s_{edge_index}")
            mdl.add_constraint(position[iv] - position[iu] == offset - n * wrap)
            mdl.add_constraint(offset <= bound + n * (1 - side))
            mdl.add_constraint(n - offset <= bound + n * side)

        mdl.minimize(bound)
        build_time = time.perf_counter() - start
        solution = mdl.solve(log_output=False)
        details = mdl.solve_details
        solve_time = float(details.time) if details is not None else (
            time.perf_counter() - start - build_time
        )

        status = _mip_status(solution, details)
        labels = None
        bandwidth = None
        if solution is not None:
            labels = {
                key: int(round(solution.get_value(position[index[key]]))) + 1
                for key in vertices
            }
            bandwidth = int(round(solution.get_objective_value()))

        lower = theoretical_lower_bound(graph)
        if details is not None and details.best_bound is not None:
            best_bound = float(details.best_bound)
            if math.isfinite(best_bound) and abs(best_bound) < 1e20:
                lower = max(lower, int(math.ceil(best_bound - 1e-6)))
        if lower > n // 2:
            lower = n // 2

        return finalize_result(
            solver_name="cplex_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=status, labels=labels, bandwidth=bandwidth,
            lower_bound=lower, build_time=build_time, solve_time=solve_time,
            message="" if solution is not None else "no solution returned",
        )

    except Exception as exc:
        return finalize_result(
            solver_name="cplex_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
        )
