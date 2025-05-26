# tests/benchmarks/benchmark_solvers.py
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.core import TSPTWProblem
from src.solvers.local_search import LocalSearchSolver, SASolver
from src.solvers.ga import GASolver
from src.solvers.alns import ALNSSolver
from src.utils import read_input_file


def run_benchmark(
    problem: TSPTWProblem, solver_class, solver_params: dict, iterations: int = 5
) -> dict:
    """Run benchmark for a single solver.

    Args:
        problem: The problem instance
        solver_class: The solver class to test
        solver_params: Parameters for the solver
        iterations: Number of iterations to run

    Returns:
        Dictionary containing benchmark results
    """
    results = []

    for i in range(iterations):
        solver = solver_class(problem, **solver_params)

        start_time = time.time()
        status = solver.solve()
        end_time = time.time()

        stats = solver.get_stats()
        results.append(
            {
                "iteration": i + 1,
                "status": status.name,
                "cost": stats["best_cost"],
                "solve_time": end_time - start_time,
                "iterations": stats["iterations"],
            }
        )

    # Calculate statistics
    df = pd.DataFrame(results)
    return {
        "solver": solver_class.__name__,
        "params": solver_params,
        "iterations": iterations,
        "success_rate": (df["status"] == "FEASIBLE").mean() * 100,
        "avg_cost": df["cost"].mean(),
        "std_cost": df["cost"].std(),
        "avg_time": df["solve_time"].mean(),
        "std_time": df["solve_time"].std(),
        "avg_iterations": df["iterations"].mean(),
        "raw_results": results,
    }


def save_benchmark_results(results: list, output_dir: str = "benchmark_results"):
    """Save benchmark results to files.

    Args:
        results: List of benchmark results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results as JSON
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Create summary DataFrame
    summary = pd.DataFrame(
        [
            {
                "solver": r["solver"],
                "success_rate": r["success_rate"],
                "avg_cost": r["avg_cost"],
                "std_cost": r["std_cost"],
                "avg_time": r["avg_time"],
                "std_time": r["std_time"],
                "avg_iterations": r["avg_iterations"],
            }
            for r in results
        ]
    )

    # Save summary as CSV
    csv_path = output_dir / f"benchmark_summary_{timestamp}.csv"
    summary.to_csv(csv_path, index=False)

    return json_path, csv_path


def plot_benchmark_results(results: list, output_dir: str = "benchmark_results"):
    """Create plots from benchmark results.

    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Create plots
    fig, axes = plt.subplots(2, 2)

    # 1. Success Rate
    success_rates = [r["success_rate"] for r in results]
    solvers = [r["solver"] for r in results]
    sns.barplot(x=solvers, y=success_rates, ax=axes[0, 0])
    axes[0, 0].set_title("Success Rate (%)")
    axes[0, 0].set_ylim(0, 100)

    # 2. Average Cost
    costs = [r["avg_cost"] for r in results]
    cost_stds = [r["std_cost"] for r in results]
    sns.barplot(x=solvers, y=costs, yerr=cost_stds, ax=axes[0, 1])
    axes[0, 1].set_title("Average Cost")

    # 3. Average Time
    times = [r["avg_time"] for r in results]
    time_stds = [r["std_time"] for r in results]
    sns.barplot(x=solvers, y=times, yerr=time_stds, ax=axes[1, 0])
    axes[1, 0].set_title("Average Solve Time (s)")

    # 4. Average Iterations
    iterations = [r["avg_iterations"] for r in results]
    sns.barplot(x=solvers, y=iterations, ax=axes[1, 1])
    axes[1, 1].set_title("Average Iterations")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = output_dir / f"benchmark_plots_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def main():
    # Define solvers and their parameters
    solvers = [
        (LocalSearchSolver, {"max_iterations": 1000}),
        (SASolver, {"initial_temp": 100.0, "cooling_rate": 0.99, "iterations": 1000}),
        (GASolver, {"population_size": 50, "generations": 100}),
        (ALNSSolver, {"iterations": 1000}),
    ]

    # Load problem instance
    problem = read_input_file("tests/data/test11/input.in")  # Using a larger test case

    # Run benchmarks
    results = []
    for solver_class, params in solvers:
        print(f"\nRunning benchmark for {solver_class.__name__}...")
        result = run_benchmark(problem, solver_class, params)
        results.append(result)

        # Print summary
        print(f"Success rate: {result['success_rate']:.1f}%")
        print(f"Average cost: {result['avg_cost']:.2f} ± {result['std_cost']:.2f}")
        print(f"Average time: {result['avg_time']:.2f}s ± {result['std_time']:.2f}s")

    # Save results
    json_path, csv_path = save_benchmark_results(results)
    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    # Create plots
    plot_path = plot_benchmark_results(results)
    print(f"  - {plot_path}")


if __name__ == "__main__":
    main()
