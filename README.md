# CBMP — SAT, CP and MIP/ILP

Solving the **Cyclic Bandwidth Minimization Problem (CBMP)** with SAT encodings
and comparing against exact CP/MIP/ILP solvers.

For a graph `G(V, E)` with `n = |V|` and a bijection `phi: V -> {1..n}`, the
cyclic bandwidth is

```
CB(G, phi) = max over {u,v} in E of min(|phi(u)-phi(v)|, n - |phi(u)-phi(v)|)
```

and the problem is to find a labeling minimizing `CB`. The decision version is
NP-complete.

## Status

- [x] Harwell-Boeing dataset (NIST collection) and graph parser
- [x] Pairwise SAT encoding (one-hot + forbidden label pairs)
- [x] Order SAT encoding (true integer-variable order encoding)
- [ ] SAT bound search (optimal B)
- [ ] Exact baselines: CPLEX CP, CPLEX MIP, Gurobi, CP-SAT
- [ ] Benchmark runner + CSV results
- [ ] Plots (cnsplots) and report

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional exact baselines and report figures:

```powershell
.venv\Scripts\python.exe -m pip install docplex cplex gurobipy ortools cnsplots
docplex config --upgrade "C:\Program Files\IBM\ILOG\CPLEX_Studio222"
```

## Encodings (`cbmp_cnf.py`)

Both build a CNF for the decision problem `CB(G) <= B`:

```python
from classes.graph import Graph
from cbmp_cnf import build_cnf, decode_model
from pysat.solvers import Solver

graph = Graph("cycle5", 5)
for u, v in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]:
    graph.add_edge(u, v)

cnf = build_cnf(graph, bound=1, encoding="order")
with Solver(name="cadical195", bootstrap_with=cnf.clauses) as solver:
    if solver.solve():
        print(decode_model(solver.get_model(), graph, encoding="order"))
```

- `pairwise`: one-hot `x[v,l]`, exactly-one per vertex and per label, and a
  clause `(-x[u,l1] or -x[v,l2])` for every edge and every label pair with
  cyclic distance `> B`.
- `order`: `o[v,k] = [label(v) >= k]` with monotonicity clauses, all-different
  via per-threshold cardinality (`sum_v o[v,k] = n-k+1`), and the cyclic bound
  expressed with three mutually exclusive regions (linear band + two wrap
  regions) guarded by selectors.

Both support `break_symmetry` (rotation) and `break_reflection`
(rotation + reflection); `amo_encoding`/`card_encoding` select the PySAT
cardinality encoding (`pairwise`, `seqcounter`, `totalizer`, ...).

## Dataset

`dataset/` contains 292 matrices from the NIST Harwell-Boeing collection,
grouped by application (`bcspwr`, `bcsstruc`, `cannes`, `dwt`, `psadmit`,
`smtape`, ...). `mtx_parser.py` parses MatrixMarket and Harwell-Boeing files
(including elemental `xxE` matrices) into simple undirected graphs; the full
shape table is in `matrix_shapes.csv`.

```powershell
.venv\Scripts\python.exe mtx_parser.py dataset
.venv\Scripts\python.exe mtx_parser.py --shapes dataset
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The encoding tests compare both SAT encodings against brute force for small
graphs, check that they agree with each other, and verify symmetry breaking.

## Layout

```
cbmp_cnf.py            SAT encodings (pairwise, order) + model decoding
mtx_parser.py          MatrixMarket / Harwell-Boeing parser -> graph
classes/graph.py       Graph structure, dataset loading and filters
classes/benchmarkresult.py
benchmark/             exact solver baselines and runner
dataset/               Harwell-Boeing matrices
matrix_shapes.csv      matrix header information
tests/                 encoding tests
report/                Elsevier cas-sc manuscript
```
