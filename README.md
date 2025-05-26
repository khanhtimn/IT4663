# TSP with Time Windows Solver

This project implements various solution approaches for the Traveling Salesman Problem with Time Windows (TSPTW).

## Problem Description

A delivery person starts from a depot (point 0) and needs to visit N customers. Each customer i:
- Has a time window [e(i), l(i)] for delivery
- Requires d(i) time units for service
- Has travel time t(i,j) from point i to j

The goal is to find the shortest route that satisfies all time window constraints.

## Project Structure

```
tsp-time-windows/
├── src/
│   ├── core/                 # Core problem definition
│   ├── solvers/             # Solution approaches
│   │   ├── alns/           # Adaptive Large Neighborhood Search
│   │   ├── ga/             # Genetic Algorithm
│   │   └── local_search/   # Local Search methods
│   └── utils/              # Utility functions
├── tests/                   # Test cases
├── examples/                # Usage examples
└── docs/                    # Documentation
```

## Solution Approaches

1. **Local Search**
   - Basic local search with 2-opt and swap moves
   - Simulated Annealing with local search

2. **Metaheuristics**
   - Genetic Algorithm with integrated operators
   - Adaptive Large Neighborhood Search with integrated operators

3. **Exact Methods**
   - Backtracking (C++ and Python implementations)

## Installation

```bash
# Clone the repository
cd tsp-time-windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from tsp_time_windows.core import TSPTWProblem
from tsp_time_windows.solvers.alns import ALNSSolver
from tsp_time_windows.solvers.ga import GASolver
from tsp_time_windows.utils import read_input_file, write_solution

# Read problem instance
problem = read_input_file("data/instance.txt")

# Using ALNS solver
alns_solver = ALNSSolver(problem, max_iterations=1000)
alns_status = alns_solver.solve()
if alns_status == ALNSSolver.Status.FEASIBLE:
    alns_solution = alns_solver.get_best_solution()
    write_solution(alns_solution)

# Using Genetic Algorithm solver
ga_solver = GASolver(problem, population_size=100, generations=1000)
ga_status = ga_solver.solve()
if ga_status == GASolver.Status.FEASIBLE:
    ga_solution = ga_solver.get_best_solution()
    write_solution(ga_solution)
```

## Input Format

```
N
e(1) l(1) d(1)
e(2) l(2) d(2)
...
e(N) l(N) d(N)
t(0,0) t(0,1) ... t(0,N)
t(1,0) t(1,1) ... t(1,N)
...
t(N,0) t(N,1) ... t(N,N)
```

## Output Format

```
N
s[1] s[2] ... s[N]
```

Where s[i] is the i-th customer in the route.

## Testing

```bash
# Run unit tests
pytest tests/unit

# Run benchmarks
pytest tests/benchmarks
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
