import random
import time
import math
from typing import List, Optional
from ...core.problem import TSPTWProblem
from ...core.solution import PermuSolution
from ..base import Solver
from ..utils.solutions import (
    create_initial_solution,
    apply_2opt_move,
    apply_swap_move,
    accept_solution_sa,
)


class SASolver(Solver):
    """Simulated Annealing solver for TSPTW."""

    def __init__(
        self,
        problem: TSPTWProblem,
        temperature: float = 100.0,
        cooling_rate: float = 0.99,
        iterations: int = 1000,
    ):
        super().__init__(problem)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.best_solution = None
        self.solve_time = 0.0

    def solve(self, **kwargs) -> Solver.Status:
        """Solve the TSPTW problem using simulated annealing."""
        start_time = time.time()

        # Initialize current solution
        current = create_initial_solution(self.problem)
        self.best_solution = current

        # Annealing loop
        for _ in range(self.iterations):
            # Generate neighbor
            neighbor = self._get_neighbor(current)

            # Accept or reject neighbor
            if accept_solution_sa(current, neighbor, self.temperature):
                current = neighbor
                if current.cost < self.best_solution.cost:
                    self.best_solution = current

            # Update temperature
            self.temperature *= self.cooling_rate

        self.solve_time = time.time() - start_time
        return (
            Solver.Status.FEASIBLE
            if self.best_solution.is_feasible()
            else Solver.Status.INFEASIBLE
        )

    def _get_neighbor(self, current: PermuSolution) -> PermuSolution:
        if random.random() < 0.5:
            # 2-opt move
            i, j = sorted(random.sample(range(1, len(current.route) - 1), 2))
            return apply_2opt_move(self.problem, current, i, j)
        else:
            # Swap move
            i, j = random.sample(range(1, len(current.route) - 1), 2)
            return apply_swap_move(self.problem, current, i, j)
