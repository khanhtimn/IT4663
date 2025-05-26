import time
from typing import List
from ...core.problem import TSPTWProblem
from ...core.solution import PermuSolution
from ..base import Solver
from ..utils.solutions import create_initial_solution, apply_2opt_move, apply_swap_move


class LocalSearchSolver(Solver):
    """Local search solver for TSPTW using 2-opt and swap moves."""

    def __init__(self, problem: TSPTWProblem, max_iterations: int = 1000):
        super().__init__(problem)
        self.iterations = max_iterations

    def solve(self, **kwargs) -> Solver.Status:
        """Solve the TSPTW problem using local search."""
        start_time = time.time()
        current = create_initial_solution(self.problem)
        self.update_best(current)

        for _ in range(self.iterations):
            improved = False

            # Try 2-opt moves
            for i in range(1, len(current.route) - 2):
                for j in range(i + 1, len(current.route) - 1):
                    new = apply_2opt_move(self.problem, current, i, j)
                    if new.is_feasible() and new.cost < current.cost:
                        current = new
                        self.update_best(current)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # Try swap moves
                for i in range(1, len(current.route) - 2):
                    for j in range(i + 1, len(current.route) - 1):
                        new = apply_swap_move(self.problem, current, i, j)
                        if new.is_feasible() and new.cost < current.cost:
                            current = new
                            self.update_best(current)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        self.solve_time = time.time() - start_time
        return (
            Solver.Status.FEASIBLE
            if self.best_solution is not None and self.best_solution.is_feasible()
            else Solver.Status.INFEASIBLE
        )
