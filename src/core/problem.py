from typing import List, Tuple
from .client import Client
from .constraints import Constraint, TimeWindowConstraint
from .solution import PermuSolution


class TSPTWProblem:
    """Traveling Salesman Problem with Time Windows."""

    def __init__(self, clients: List[Client], start_at: Client):
        self.clients = clients
        self.start = start_at
        self.constraints: List[Constraint] = []
        self.add_constraint(TimeWindowConstraint())

        # Validate problem instance
        from ..utils.validation import validate_problem, validate_constraints

        errors = validate_problem(self)
        if errors:
            raise ValueError(f"Invalid problem instance:\n" + "\n".join(errors))

    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the problem."""
        if constraint is None:
            raise ValueError("Cannot add None constraint")
        self.constraints.append(constraint)

        # Validate constraints after adding
        from ..utils.validation import validate_constraints

        errors = validate_constraints(self)
        if errors:
            self.constraints.pop()  # Remove the invalid constraint
            raise ValueError(f"Invalid constraint:\n" + "\n".join(errors))

    def cal_violations(self, solution: PermuSolution) -> int:
        """Calculate total number of constraint violations."""
        total_violations = 0
        for constraint in self.constraints:
            total_violations += constraint.get_violation(solution)
        return total_violations

    def cal_penalty(self, solution: PermuSolution) -> float:
        """Calculate total penalty for constraint violations."""
        total_penalties = 0
        for constraint in self.constraints:
            total_penalties += constraint.get_penalty(solution)
        return total_penalties

    def check(self, solution: PermuSolution) -> bool:
        """Check if solution satisfies all constraints."""
        return self.cal_violations(solution) == 0

    def cal_cost(self, solution: PermuSolution) -> float:
        """Calculate total travel cost of the route."""
        route = solution.decode()
        if not route:
            return float("inf")

        total_cost = 0.0
        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            total_cost += cur.travel_times.get(nxt.id, float("inf"))
        return total_cost
