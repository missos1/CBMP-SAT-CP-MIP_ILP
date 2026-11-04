from dataclasses import dataclass
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