from abc import ABC, abstractmethod
from typing import List
from .solution import PermuSolution


class Constraint(ABC):
    """Base class for all constraints in the TSPTW problem."""

    @abstractmethod
    def get_violation(self, solution: PermuSolution) -> int:
        """Calculate the number of violations in the solution.

        Args:
            solution: The solution to check

        Returns:
            int: Number of violations
        """
        pass

    def get_penalty(self, solution: PermuSolution) -> float:
        """Calculate the penalty for constraint violations.

        Args:
            solution: The solution to check

        Returns:
            float: Penalty value
        """
        return self.get_violation(solution)

    def check(self, solution: PermuSolution) -> bool:
        """Check if the solution satisfies the constraint.

        Args:
            solution: The solution to check

        Returns:
            bool: True if no violations, False otherwise
        """
        return self.get_violation(solution) == 0


class TimeWindowConstraint(Constraint):
    """Constraint for time windows in TSPTW."""

    def get_violation(self, solution: PermuSolution) -> int:
        """Calculate time window violations.

        Args:
            solution: The solution to check

        Returns:
            int: Number of time window violations
        """
        route = solution.decode()
        curr_time = 0
        violations = 0

        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times.get(nxt.id, float("inf"))
            if travel_time == float("inf"):
                return float("inf")

            arrival_time = curr_time + travel_time

            if arrival_time < nxt.earliness:
                arrival_time = nxt.earliness
            if arrival_time > nxt.tardiness:
                violations += 1

            curr_time = arrival_time + nxt.service_time

        return violations

    def get_penalty(self, solution: PermuSolution) -> float:
        """Calculate time window violation penalties.

        Args:
            solution: The solution to check

        Returns:
            float: Total penalty for time window violations
        """
        route = solution.decode()
        curr_time = 0
        penalty = 0.0

        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times.get(nxt.id, float("inf"))
            if travel_time == float("inf"):
                return float("inf")

            arrival_time = curr_time + travel_time

            if arrival_time < nxt.earliness:
                penalty += nxt.earliness - arrival_time
                arrival_time = nxt.earliness
            if arrival_time > nxt.tardiness:
                penalty += 3 * (arrival_time - nxt.tardiness)

            curr_time = arrival_time + nxt.service_time

        return penalty


class MustFollowConstraint(Constraint):
    """Constraint ensuring certain clients must be visited before others."""

    def get_violation(self, solution: PermuSolution) -> int:
        """Calculate violations of the must-follow constraint.

        Args:
            solution: The solution to check

        Returns:
            int: Number of must-follow violations
        """
        violations = 0
        route = solution.decode()

        for i in range(1, solution.size - 1):
            for j in range(1, solution.size - 1):
                if (
                    i > j
                    and route[i].tardiness + route[i].travel_times[route[j].id]
                    < route[j].earliness
                ):
                    violations += 1

        return violations


class CapacityConstraint(Constraint):
    """Constraint for vehicle capacity in TSPTW."""

    def __init__(self, max_capacity: float):
        """Initialize capacity constraint.

        Args:
            max_capacity: Maximum vehicle capacity
        """
        self.max_capacity = max_capacity

    def get_violation(self, solution: PermuSolution) -> int:
        """Calculate capacity violations.

        Args:
            solution: The solution to check

        Returns:
            int: Number of capacity violations
        """
        route = solution.decode()
        curr_load = 0
        violations = 0

        for i in range(1, solution.size - 1):  # Skip depot at start/end
            curr_load += route[i].service_time
            if curr_load > self.max_capacity:
                violations += 1

        return violations

    def get_penalty(self, solution: PermuSolution) -> float:
        """Calculate capacity violation penalties.

        Args:
            solution: The solution to check

        Returns:
            float: Total penalty for capacity violations
        """
        route = solution.decode()
        curr_load = 0
        penalty = 0.0

        for i in range(1, solution.size - 1):  # Skip depot at start/end
            curr_load += route[i].service_time
            if curr_load > self.max_capacity:
                penalty += 2 * (curr_load - self.max_capacity)

        return penalty


class ServiceTimeConstraint(Constraint):
    """Constraint for maximum service time at each location."""

    def __init__(self, max_service_time: float):
        """Initialize service time constraint.

        Args:
            max_service_time: Maximum allowed service time at any location
        """
        self.max_service_time = max_service_time

    def get_violation(self, solution: PermuSolution) -> int:
        """Calculate service time violations.

        Args:
            solution: The solution to check

        Returns:
            int: Number of service time violations
        """
        route = solution.decode()
        violations = 0

        for i in range(1, solution.size - 1):  # Skip depot at start/end
            if route[i].service_time > self.max_service_time:
                violations += 1

        return violations

    def get_penalty(self, solution: PermuSolution) -> float:
        """Calculate service time violation penalties.

        Args:
            solution: The solution to check

        Returns:
            float: Total penalty for service time violations
        """
        route = solution.decode()
        penalty = 0.0

        for i in range(1, solution.size - 1):  # Skip depot at start/end
            if route[i].service_time > self.max_service_time:
                penalty += 1.5 * (route[i].service_time - self.max_service_time)

        return penalty


class RouteLengthConstraint(Constraint):
    """Constraint for maximum route length."""

    def __init__(self, max_length: float):
        """Initialize route length constraint.

        Args:
            max_length: Maximum allowed route length
        """
        self.max_length = max_length

    def get_violation(self, solution: PermuSolution) -> int:
        """Calculate route length violations.

        Args:
            solution: The solution to check

        Returns:
            int: Number of route length violations
        """
        route = solution.decode()
        total_length = 0

        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times.get(nxt.id, float("inf"))
            if travel_time == float("inf"):
                return 1
            total_length += travel_time

        return 1 if total_length > self.max_length else 0

    def get_penalty(self, solution: PermuSolution) -> float:
        """Calculate route length violation penalties.

        Args:
            solution: The solution to check

        Returns:
            float: Total penalty for route length violations
        """
        route = solution.decode()
        total_length = 0

        for i in range(solution.size - 1):
            cur, nxt = route[i], route[i + 1]
            travel_time = cur.travel_times.get(nxt.id, float("inf"))
            if travel_time == float("inf"):
                return float("inf")
            total_length += travel_time

        return (
            2 * (total_length - self.max_length)
            if total_length > self.max_length
            else 0.0
        )
