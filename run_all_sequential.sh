#!/usr/bin/env bash
set -e

PY=""

echo "=== STARTING CBMP BENCHMARK SUITE ==="
date

# 1. SAT Solvers (Order, Pairwise, Incremental)
echo ">>> [1/7] Running sat_order..."
$PY run_benchmarks.py --solver sat_order

echo ">>> [2/7] Running sat_pairwise..."
$PY run_benchmarks.py --solver sat_pairwise

echo ">>> [3/7] Running sat_incremental..."
$PY run_benchmarks.py --solver sat_incremental

# 2. Gurobi MIP
echo ">>> [4/7] Running gurobi..."
$PY run_benchmarks.py --solver gurobi

# 3. HiGHS MIP
echo ">>> [5/7] Running highs..."
$PY run_benchmarks.py --solver highs

# 4. OR-Tools CP-SAT
echo ">>> [6/7] Running ortools..."
$PY run_benchmarks.py --solver ortools

# 5. CPLEX (CP and MIP)
echo ">>> [7/7] Running cplex_both..."
$PY run_benchmarks.py --solver cplex_both

echo "=== ALL BENCHMARKS COMPLETED SUCCESSFULLY ==="
date
