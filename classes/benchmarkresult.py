from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkResult:
    solver_name: str
    graph_name: str
    vertex_count: int
    edge_count: int
    symmetry_breaking: bool
    reflection_breaking: bool
    encoding: Optional[str] = None
    build_time: Optional[float] = None
    solve_time: Optional[float] = None
    total_time: Optional[float] = None
    bandwidth: Optional[int] = None
    status: Optional[str] = None
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    optimal: Optional[bool] = None
    solution: Optional[dict] = None
    verified: Optional[bool] = None
    message: Optional[str] = None
    extra: dict = field(default_factory=dict)
