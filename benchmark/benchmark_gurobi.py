"""Gurobi MIP baseline for CBMP (compact assignment formulation)."""

import math
import time

import gurobipy as gp
from gurobipy import GRB

from benchmark.common import (
    ERROR,
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    TIMEOUT,
    finalize_result,
    theoretical_lower_bound,
)
from classes.graph import Graph


def _map_status(model) -> str:
    if model.Status == GRB.OPTIMAL:
        return OPTIMAL
    if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return INFEASIBLE
    if model.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED):
        return FEASIBLE if model.SolCount > 0 else TIMEOUT
    return FEASIBLE if model.SolCount > 0 else ERROR


def benchmap_gurobi_mip(graph: Graph, break_symmetry: bool = False,
                        break_reflection: bool = False,
                        time_limit: float = 60.0,
                        mip_gap: float = 0.0) -> object:
    n = graph.get_vertex_count()
    edges = graph.get_edges()

    if n == 0 or not edges:
        labels = {v: i for i, v in enumerate(graph.adjacency_list, 1)}
        return finalize_result(
            solver_name="gurobi_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=OPTIMAL, labels=labels, bandwidth=0,
            lower_bound=0, upper_bound=0,
        )

    vertices = list(graph.adjacency_list)
    index = {v: i for i, v in enumerate(vertices)}
    start = time.perf_counter()

    try:
        model = gp.Model("cbmp_gurobi")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(time_limit)
        model.Params.MIPGap = float(mip_gap)

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, l] for l in range(n)) == 1)
        for l in range(n):
            model.addConstr(gp.quicksum(x[i, l] for i in range(n)) == 1)

        if break_symmetry:
            model.addConstr(x[0, 0] == 1)
            if break_reflection and n > 2:
                neighbours = graph.get_neighbors(vertices[0])
                target = index[neighbours[0]] if neighbours else 1
                model.addConstr(
                    gp.quicksum(x[target, l] for l in range(n // 2 + 1, n)) == 0
                )

        bound = model.addVar(lb=0, ub=n // 2, vtype=GRB.INTEGER, name="B")
        position = model.addVars(n, lb=0, ub=n - 1, vtype=GRB.INTEGER, name="p")
        for i in range(n):
            model.addConstr(
                position[i] == gp.quicksum(l * x[i, l] for l in range(n))
            )

        for edge_index, (u, v) in enumerate(edges):
            iu, iv = index[u], index[v]
            offset = model.addVar(lb=0, ub=n - 1, vtype=GRB.INTEGER,
                                  name=f"d_{edge_index}")
            wrap = model.addVar(vtype=GRB.BINARY, name=f"w_{edge_index}")
            side = model.addVar(vtype=GRB.BINARY, name=f"s_{edge_index}")
            model.addConstr(position[iv] - position[iu] == offset - n * wrap)
            model.addConstr(offset <= bound + n * (1 - side))
            model.addConstr(n - offset <= bound + n * side)

        model.setObjective(bound, GRB.MINIMIZE)
        build_time = time.perf_counter() - start
        model.optimize()
        solve_time = float(model.Runtime)

        status = _map_status(model)
        labels = None
        if model.SolCount > 0:
            labels = {}
            for i, key in enumerate(vertices):
                for l in range(n):
                    if x[i, l].X > 0.5:
                        labels[key] = l + 1
                        break

        lower = theoretical_lower_bound(graph)
        if math.isfinite(model.ObjBound) and abs(model.ObjBound) < 1e20:
            lower = max(lower, int(math.ceil(model.ObjBound - 1e-6)))
        if lower > n // 2:
            lower = n // 2

        return finalize_result(
            solver_name="gurobi_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=status, labels=labels,
            lower_bound=lower,
            upper_bound=int(round(model.ObjVal)) if model.SolCount > 0 else None,
            build_time=build_time, solve_time=solve_time,
            extra={
                "sol_count": int(model.SolCount),
                "gurobi_status": int(model.Status),
                "objective_bound": float(model.ObjBound),
            },
        )

    except Exception as exc:
        return finalize_result(
            solver_name="gurobi_mip", graph=graph,
            symmetry_breaking=break_symmetry,
            reflection_breaking=break_reflection,
            status=ERROR, build_time=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
        )
