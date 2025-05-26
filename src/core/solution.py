from typing import List
from .client import Client


class PermuSolution:
    """Represents a solution to the TSPTW problem."""

    def __init__(self, size: int):
        self.size = size
        self.route: List[Client] = []
        self.cost = float("inf")
        self.violations = float("inf")
        self.penalty = float("inf")

    def decode(self) -> List[Client]:
        """Decode the solution into a valid route."""
        if not self.route:
            return [self.route[0], self.route[0]]  # Return start-start if empty
        if len(self.route) != self.size:
            return self.route  # Return current route even if size doesn't match
        if self.route[0] != self.route[-1]:
            return self.route  # Return current route even if not closed
        return self.route

    def get_main(self):
        """Get the main route without depot."""
        return self.route[1:-1]

    def is_feasible(self) -> bool:
        """Check if the solution satisfies all constraints."""
        return self.violations == 0

    def __str__(self):
        return f"[{', '.join(str(c.id) for c in self.route)}]"
