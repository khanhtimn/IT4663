from src.core import TSPTWProblem
from src.solvers.local_search import LocalSearchSolver, SASolver
from src.solvers.ga import GASolver
from src.solvers.alns import ALNSSolver
from src.utils import read_input_file, write_solution


def main():
    # Read problem instance from file
    problem = read_input_file("data/instance.txt")

    # Choose a solver (uncomment the one you want to use)
    solver = LocalSearchSolver(problem, max_iterations=1000)
    # solver = SASolver(problem, initial_temp=100.0, cooling_rate=0.99, iterations=1000)
    # solver = GASolver(problem, population_size=50, generations=100)
    # solver = ALNSSolver(problem, iterations=1000)

    # Solve
    status = solver.solve()

    # Get results
    if status == solver.Status.FEASIBLE:
        solution = solver.get_best_solution()
        stats = solver.get_stats()
        print(f"Found feasible solution!")
        print(f"Cost: {stats['best_cost']}")
        print(f"Solve time: {stats['solve_time']:.2f}s")
        print(f"Iterations: {stats['iterations']}")
        write_solution(solution)
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
