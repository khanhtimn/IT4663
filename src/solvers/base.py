from abc import ABC, abstractmethod
from typing import Optional
from ..core.problem import TSPTWProblem
from ..core.solution import PermuSolution


class Solver(ABC):
    """Abstract base class for all TSPTW solvers."""

    class Status:
        """Solver status codes."""

        FEASIBLE = "FEASIBLE"
        INFEASIBLE = "INFEASIBLE"
        TIMEOUT = "TIMEOUT"
        ERROR = "ERROR"

    def __init__(self, problem: TSPTWProblem):
        self.problem = problem
        self.best_solution: Optional[PermuSolution] = None
        self.best_cost = float("inf")
        self.best_violations = float("inf")
        self.best_penalty = float("inf")
        self.solve_time = 0.0
        self.iterations = 0

    @abstractmethod
    def solve(self, **kwargs) -> Status:
        """Solve the TSPTW problem.

        Returns:
            Status: The status of the solution attempt.
        """
        pass

    def update_best(self, solution: PermuSolution) -> bool:
        """Update the best solution if the new one is better.

        Args:
            solution: The new solution to evaluate.

        Returns:
            bool: True if the best solution was updated.
        """
        if not solution.is_feasible():
            return False

        if solution.cost < self.best_cost:
            self.best_solution = solution
            self.best_cost = solution.cost
            self.best_violations = solution.violations
            self.best_penalty = solution.penalty
            return True

        return False

    def get_best_solution(self) -> Optional[PermuSolution]:
        """Get the best solution found so far."""
        return self.best_solution

    def get_stats(self) -> dict:
        """Get solver statistics."""
        return {
            "solve_time": self.solve_time,
            "iterations": self.iterations,
            "best_cost": self.best_cost,
            "best_violations": self.best_violations,
            "best_penalty": self.best_penalty,
        }
