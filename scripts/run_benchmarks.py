#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from src.core import TSPTWProblem
from src.solvers.local_search.basic import LocalSearchSolver
from src.solvers.local_search.sa import SASolver
from src.solvers.metaheuristic.ga import GASolver
from src.solvers.metaheuristic.alns import ALNSSolver
from src.solvers.backtracking import BacktrackingSolver
from src.utils import read_input_file, validate_problem, validate_solution


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_solver_instance(solver_name, problem, **kwargs):
    """Create a solver instance based on the solver name."""
    solvers = {
        "local_search": LocalSearchSolver,
        "sa": SASolver,
        "ga": GASolver,
        "alns": ALNSSolver,
        "backtracking": BacktrackingSolver,
    }

    if solver_name not in solvers:
        raise ValueError(f"Unknown solver: {solver_name}")

    return solvers[solver_name](problem, **kwargs)


def run_benchmark(solver_name, problem, num_iterations=1, **solver_kwargs):
    """Run a single benchmark for a solver."""
    results = []

    for i in range(num_iterations):
        solver = get_solver_instance(solver_name, problem, **solver_kwargs)

        start_time = time.time()
        status = solver.solve()
        end_time = time.time()

        stats = solver.get_stats()
        best_solution = solver.get_best_solution()

        # Validate solution if one was found
        validation_errors = []
        if best_solution is not None:
            validation_errors = validate_solution(best_solution, problem)

        # Calculate additional metrics
        metrics = {
            "iteration": i + 1,
            "status": status,
            "cost": stats["best_cost"],
            "solve_time": end_time - start_time,
            "iterations": stats["iterations"],
            "validation_errors": validation_errors,
            "solver": solver_name,
            "feasible": best_solution.is_feasible() if best_solution else False,
            "violations": best_solution.violations if best_solution else float("inf"),
            "penalty": best_solution.penalty if best_solution else float("inf"),
            "route_length": len(best_solution.route) if best_solution else 0,
            "memory_usage": sys.getsizeof(best_solution) if best_solution else 0,
        }

        # Add solver-specific metrics
        if solver_name == "ga":
            metrics.update(
                {
                    "population_size": stats.get("population_size", 0),
                    "generations": stats.get("generations", 0),
                    "mutation_rate": stats.get("mutation_rate", 0),
                    "crossover_rate": stats.get("crossover_rate", 0),
                }
            )
        elif solver_name == "sa":
            metrics.update(
                {
                    "initial_temperature": stats.get("initial_temperature", 0),
                    "final_temperature": stats.get("final_temperature", 0),
                    "cooling_rate": stats.get("cooling_rate", 0),
                }
            )
        elif solver_name == "alns":
            metrics.update(
                {
                    "destroy_weights": stats.get("destroy_weights", []),
                    "repair_weights": stats.get("repair_weights", []),
                    "segment_size": stats.get("segment_size", 0),
                }
            )

        results.append(metrics)

    return results


def save_results(results, output_dir):
    """Save benchmark results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group results by solver
    solver_results = {}
    for result in results:
        solver = result["solver"]
        if solver not in solver_results:
            solver_results[solver] = []
        solver_results[solver].append(result)

    # Save results for each solver in its own directory
    for solver, solver_data in solver_results.items():
        solver_dir = output_dir / solver
        solver_dir.mkdir(exist_ok=True)

        # Save CSV
        csv_path = solver_dir / f"results_{timestamp}.csv"
        pd.DataFrame(solver_data).to_csv(csv_path, index=False)

        # Create solver-specific report
        report_path = solver_dir / f"report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(f"{solver.upper()} Solver Report\n")
            f.write("=" * 80 + "\n\n")

            df = pd.DataFrame(solver_data)
            f.write(f"Total runs: {len(df)}\n")
            f.write(
                f"Feasible solutions: {df['feasible'].sum()} ({df['feasible'].mean() * 100:.1f}%)\n\n"
            )
            f.write(f"Mean cost: {df['cost'].mean():.2f} ± {df['cost'].std():.2f}\n")
            f.write(
                f"Mean solve time: {df['solve_time'].mean():.2f}s ± {df['solve_time'].std():.2f}s\n"
            )
            f.write(
                f"Mean violations: {df['violations'].mean():.2f} ± {df['violations'].std():.2f}\n"
            )
            f.write(
                f"Mean penalty: {df['penalty'].mean():.2f} ± {df['penalty'].std():.2f}\n"
            )
            f.write(
                f"Mean iterations: {df['iterations'].mean():.2f} ± {df['iterations'].std():.2f}\n"
            )

        print(f"\nResults for {solver} saved to:")
        print(f"  - {csv_path}")
        print(f"  - {report_path}")

    # Create overall comparison report
    comparison_report = output_dir / f"comparison_report_{timestamp}.txt"
    with open(comparison_report, "w") as f:
        f.write("Solver Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Overall Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Total solvers: {len(solver_results)}\n")
        f.write(
            f"Total feasible solutions: {sum(r['feasible'] for r in results)} ({sum(r['feasible'] for r in results) / len(results) * 100:.1f}%)\n\n"
        )

        f.write("Solver Comparison\n")
        f.write("-" * 80 + "\n")
        for solver, solver_data in solver_results.items():
            df = pd.DataFrame(solver_data)
            f.write(f"\n{solver.upper()} Solver\n")
            f.write("-" * 40 + "\n")
            f.write(f"Runs: {len(df)}\n")
            f.write(
                f"Feasible solutions: {df['feasible'].sum()} ({df['feasible'].mean() * 100:.1f}%)\n"
            )
            f.write(f"Mean cost: {df['cost'].mean():.2f} ± {df['cost'].std():.2f}\n")
            f.write(
                f"Mean solve time: {df['solve_time'].mean():.2f}s ± {df['solve_time'].std():.2f}s\n"
            )
            f.write(
                f"Mean violations: {df['violations'].mean():.2f} ± {df['violations'].std():.2f}\n"
            )
            f.write(
                f"Mean penalty: {df['penalty'].mean():.2f} ± {df['penalty'].std():.2f}\n"
            )

    print(f"\nComparison report saved to:")
    print(f"  - {comparison_report}")


def create_plots(results, output_dir):
    """Create and save plots from benchmark results."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group results by solver
    solver_results = {}
    for result in results:
        solver = result["solver"]
        if solver not in solver_results:
            solver_results[solver] = []
        solver_results[solver].append(result)

    # Create solver-specific plots
    for solver, solver_data in solver_results.items():
        solver_dir = output_dir / solver
        solver_dir.mkdir(exist_ok=True)

        df = pd.DataFrame(solver_data)

        # Create box plots for each metric
        metrics = ["cost", "solve_time", "violations", "penalty", "iterations"]
        for metric in metrics:
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    y=df[metric],
                    name=metric,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )
            fig.update_layout(
                title=f"{metric.replace('_', ' ').title()} Distribution",
                yaxis_title=metric.replace("_", " ").title(),
                showlegend=False,
            )
            fig.write_html(solver_dir / f"{metric}_distribution_{timestamp}.html")

        print(f"\nPlots for {solver} saved to:")
        for metric in metrics:
            print(f"  - {solver_dir}/{metric}_distribution_{timestamp}.html")

    # Create comparison plots in root directory
    # 1. Success Rate Comparison
    success_rates = [
        (pd.DataFrame(data)["status"] == "FEASIBLE").mean() * 100
        for data in solver_results.values()
    ]
    solvers = list(solver_results.keys())

    fig1 = px.bar(
        x=solvers,
        y=success_rates,
        title="Success Rate Comparison",
        labels={"x": "Solver", "y": "Success Rate (%)"},
    )
    fig1.write_html(output_dir / f"success_rate_comparison_{timestamp}.html")

    # 2. Cost Comparison
    fig2 = go.Figure()
    for solver, data in solver_results.items():
        df = pd.DataFrame(data)
        fig2.add_trace(
            go.Box(
                y=df["cost"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig2.update_layout(
        title="Cost Comparison",
        yaxis_title="Cost",
        showlegend=True,
    )
    fig2.write_html(output_dir / f"cost_comparison_{timestamp}.html")

    # 3. Time Comparison
    fig3 = go.Figure()
    for solver, data in solver_results.items():
        df = pd.DataFrame(data)
        fig3.add_trace(
            go.Box(
                y=df["solve_time"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig3.update_layout(
        title="Solve Time Comparison",
        yaxis_title="Time (s)",
        showlegend=True,
    )
    fig3.write_html(output_dir / f"time_comparison_{timestamp}.html")

    # 4. Violations Comparison
    fig4 = go.Figure()
    for solver, data in solver_results.items():
        df = pd.DataFrame(data)
        fig4.add_trace(
            go.Box(
                y=df["violations"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig4.update_layout(
        title="Violations Comparison",
        yaxis_title="Violations",
        showlegend=True,
    )
    fig4.write_html(output_dir / f"violations_comparison_{timestamp}.html")

    # 5. Penalty Comparison
    fig5 = go.Figure()
    for solver, data in solver_results.items():
        df = pd.DataFrame(data)
        fig5.add_trace(
            go.Box(
                y=df["penalty"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig5.update_layout(
        title="Penalty Comparison",
        yaxis_title="Penalty",
        showlegend=True,
    )
    fig5.write_html(output_dir / f"penalty_comparison_{timestamp}.html")

    # 6. Iterations Comparison
    fig6 = go.Figure()
    for solver, data in solver_results.items():
        df = pd.DataFrame(data)
        fig6.add_trace(
            go.Box(
                y=df["iterations"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig6.update_layout(
        title="Iterations Comparison",
        yaxis_title="Iterations",
        showlegend=True,
    )
    fig6.write_html(output_dir / f"iterations_comparison_{timestamp}.html")

    # 7. Parallel Coordinates Plot
    parallel_data = []
    for solver, data in solver_results.items():
        df = pd.DataFrame(data)
        solver_data = {
            "solver": solver,
            "cost": df["cost"].mean(),
            "solve_time": df["solve_time"].mean(),
            "violations": df["violations"].mean(),
            "penalty": df["penalty"].mean(),
            "iterations": df["iterations"].mean(),
        }
        parallel_data.append(solver_data)

    fig7 = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=[i for i in range(len(solvers))],
                colorscale="Viridis",
            ),
            dimensions=[
                dict(
                    range=[0, len(solvers) - 1],
                    ticktext=solvers,
                    tickvals=list(range(len(solvers))),
                    label="Solver",
                    values=[i for i in range(len(solvers))],
                ),
                dict(
                    range=[
                        min(d["cost"] for d in parallel_data),
                        max(d["cost"] for d in parallel_data),
                    ],
                    label="Cost",
                    values=[d["cost"] for d in parallel_data],
                ),
                dict(
                    range=[
                        min(d["solve_time"] for d in parallel_data),
                        max(d["solve_time"] for d in parallel_data),
                    ],
                    label="Solve Time",
                    values=[d["solve_time"] for d in parallel_data],
                ),
                dict(
                    range=[
                        min(d["violations"] for d in parallel_data),
                        max(d["violations"] for d in parallel_data),
                    ],
                    label="Violations",
                    values=[d["violations"] for d in parallel_data],
                ),
                dict(
                    range=[
                        min(d["penalty"] for d in parallel_data),
                        max(d["penalty"] for d in parallel_data),
                    ],
                    label="Penalty",
                    values=[d["penalty"] for d in parallel_data],
                ),
                dict(
                    range=[
                        min(d["iterations"] for d in parallel_data),
                        max(d["iterations"] for d in parallel_data),
                    ],
                    label="Iterations",
                    values=[d["iterations"] for d in parallel_data],
                ),
            ],
        )
    )
    fig7.update_layout(title="Solver Comparison (Parallel Coordinates)")
    fig7.write_html(output_dir / f"solver_comparison_parallel_{timestamp}.html")

    # 8. Radar Chart
    def normalize(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) if max_val > min_val else 0.5

    radar_data = []
    for d in parallel_data:
        normalized_data = {
            "solver": d["solver"],
            "cost": normalize(
                d["cost"],
                min(d["cost"] for d in parallel_data),
                max(d["cost"] for d in parallel_data),
            ),
            "solve_time": normalize(
                d["solve_time"],
                min(d["solve_time"] for d in parallel_data),
                max(d["solve_time"] for d in parallel_data),
            ),
            "violations": normalize(
                d["violations"],
                min(d["violations"] for d in parallel_data),
                max(d["violations"] for d in parallel_data),
            ),
            "penalty": normalize(
                d["penalty"],
                min(d["penalty"] for d in parallel_data),
                max(d["penalty"] for d in parallel_data),
            ),
            "iterations": normalize(
                d["iterations"],
                min(d["iterations"] for d in parallel_data),
                max(d["iterations"] for d in parallel_data),
            ),
        }
        radar_data.append(normalized_data)

    fig8 = go.Figure()
    for d in radar_data:
        fig8.add_trace(
            go.Scatterpolar(
                r=[
                    d["cost"],
                    d["solve_time"],
                    d["violations"],
                    d["penalty"],
                    d["iterations"],
                ],
                theta=["Cost", "Solve Time", "Violations", "Penalty", "Iterations"],
                name=d["solver"],
                fill="toself",
            )
        )
    fig8.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        title="Solver Comparison (Radar Chart)",
        showlegend=True,
    )
    fig8.write_html(output_dir / f"solver_comparison_radar_{timestamp}.html")

    print(f"\nComparison plots saved to:")
    print(f"  - {output_dir}/success_rate_comparison_{timestamp}.html")
    print(f"  - {output_dir}/cost_comparison_{timestamp}.html")
    print(f"  - {output_dir}/time_comparison_{timestamp}.html")
    print(f"  - {output_dir}/violations_comparison_{timestamp}.html")
    print(f"  - {output_dir}/penalty_comparison_{timestamp}.html")
    print(f"  - {output_dir}/iterations_comparison_{timestamp}.html")
    print(f"  - {output_dir}/solver_comparison_parallel_{timestamp}.html")
    print(f"  - {output_dir}/solver_comparison_radar_{timestamp}.html")


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for TSPTW solvers")
    parser.add_argument(
        "--instance", required=True, help="Path to the problem instance file"
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["local_search", "sa", "ga", "alns", "backtracking"],
        help="List of solvers to benchmark",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per solver"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Time limit in seconds for each solver run",
    )
    args = parser.parse_args()

    # Read and validate problem instance
    try:
        problem = read_input_file(args.instance)
        errors = validate_problem(problem)
        if errors:
            print("Problem validation errors:")
            for error in errors:
                print(f"- {error}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading problem instance: {e}")
        sys.exit(1)

    # Default solver parameters
    solver_params = {
        "local_search": {"max_iterations": 1000},
        "sa": {
            "temperature": 100.0,
            "cooling_rate": 0.99,
            "iterations": 1000,
        },
        "ga": {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elite_size": 5,
            "tournament_size": 3,
        },
        "alns": {
            "iterations": 1000,
            "segment_size": 3,
            "temperature": 100.0,
            "cooling_rate": 0.99,
            "weight_update_interval": 100,
        },
        "backtracking": {"max_time": args.time_limit},
    }

    # Run benchmarks
    all_results = []
    for solver_name in args.solvers:
        print(f"\nRunning benchmark for {solver_name}...")
        results = run_benchmark(
            solver_name,
            problem,
            num_iterations=args.iterations,
            **solver_params.get(solver_name, {}),
        )

        # Add solver name to results
        for r in results:
            r["solver"] = solver_name

        all_results.extend(results)

        # Print summary
        df = pd.DataFrame(results)
        print(f"\nSummary for {solver_name}:")
        print(f"  Average cost: {df['cost'].mean():.2f} ± {df['cost'].std():.2f}")
        print(
            f"  Average solve time: {df['solve_time'].mean():.2f}s ± {df['solve_time'].std():.2f}s"
        )
        print(f"  Success rate: {(df['status'] == 'FEASIBLE').mean() * 100:.1f}%")
        print(
            f"  Average violations: {df['violations'].mean():.2f} ± {df['violations'].std():.2f}"
        )
        print(
            f"  Average penalty: {df['penalty'].mean():.2f} ± {df['penalty'].std():.2f}"
        )

        # Print validation errors if any
        validation_errors = [
            r["validation_errors"] for r in results if r["validation_errors"]
        ]
        if validation_errors:
            print("  Validation errors found:")
            for errors in validation_errors:
                for error in errors:
                    print(f"    - {error}")

    # Save results
    save_results(all_results, args.output_dir)

    # Create plots
    create_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
