from typing import List, Optional, TYPE_CHECKING
from ..core.constraints import (
    Constraint,
    TimeWindowConstraint,
    MustFollowConstraint,
    CapacityConstraint,
    ServiceTimeConstraint,
    RouteLengthConstraint,
)

if TYPE_CHECKING:
    from ..core.problem import TSPTWProblem
    from ..core.solution import PermuSolution
    from ..core.client import Client


def validate_solution(solution: "PermuSolution", problem: "TSPTWProblem") -> List[str]:
    """Validate a solution against the problem instance.

    Args:
        solution: The solution to validate
        problem: The problem instance

    Returns:
        List[str]: List of validation error messages, empty if solution is valid
    """
    errors = []

    # Check if solution exists
    if solution is None:
        return ["Solution is None"]

    # Check route length
    if len(solution.route) != solution.size:
        errors.append(
            f"Route length {len(solution.route)} does not match solution size {solution.size}"
        )

    # Check if route starts and ends at depot
    if solution.route[0] != problem.start or solution.route[-1] != problem.start:
        errors.append("Route must start and end at depot")

    # Check if all clients are visited exactly once
    client_ids = set(c.id for c in solution.route[1:-1])  # Exclude depot
    expected_ids = set(c.id for c in problem.clients[1:])  # Exclude depot
    if client_ids != expected_ids:
        missing = expected_ids - client_ids
        extra = client_ids - expected_ids
        if missing:
            errors.append(f"Missing clients: {missing}")
        if extra:
            errors.append(f"Extra clients: {extra}")

    # Check for invalid travel times and time windows
    curr_time = 0
    curr_load = 0
    total_length = 0

    for i in range(len(solution.route) - 1):
        cur, nxt = solution.route[i], solution.route[i + 1]

        # Check travel time
        if nxt.id not in cur.travel_times:
            errors.append(f"Missing travel time from {cur.id} to {nxt.id}")
            continue
        travel_time = cur.travel_times[nxt.id]
        if travel_time < 0:
            errors.append(f"Negative travel time from {cur.id} to {nxt.id}")
            continue
        total_length += travel_time

        # Update time and check time windows
        arrival_time = curr_time + travel_time
        if arrival_time < nxt.earliness:
            errors.append(
                f"Early arrival at client {nxt.id}: {arrival_time} < {nxt.earliness}"
            )
        if arrival_time > nxt.tardiness:
            errors.append(
                f"Late arrival at client {nxt.id}: {arrival_time} > {nxt.tardiness}"
            )

        # Update current time considering waiting
        curr_time = max(arrival_time, nxt.earliness) + nxt.service_time

        # Update load and check capacity
        if i > 0:  # Skip depot
            curr_load += cur.service_time
            for constraint in problem.constraints:
                if isinstance(constraint, CapacityConstraint):
                    if curr_load > constraint.max_capacity:
                        errors.append(
                            f"Capacity exceeded at client {cur.id}: {curr_load} > {constraint.max_capacity}"
                        )

    # Check route length constraint
    for constraint in problem.constraints:
        if isinstance(constraint, RouteLengthConstraint):
            if total_length > constraint.max_length:
                errors.append(
                    f"Route length exceeded: {total_length} > {constraint.max_length}"
                )

    # Check service time constraints
    for constraint in problem.constraints:
        if isinstance(constraint, ServiceTimeConstraint):
            for client in solution.route[1:-1]:  # Skip depot
                if client.service_time > constraint.max_service_time:
                    errors.append(
                        f"Service time exceeded at client {client.id}: {client.service_time} > {constraint.max_service_time}"
                    )

    # Check must-follow constraints
    for constraint in problem.constraints:
        if isinstance(constraint, MustFollowConstraint):
            violations = constraint.get_violation(solution)
            if violations > 0:
                errors.append(f"Must-follow constraint violations: {violations}")

    return errors


def validate_problem(problem: "TSPTWProblem") -> List[str]:
    """Validate a problem instance.

    Args:
        problem: The problem instance to validate

    Returns:
        List[str]: List of validation error messages, empty if problem is valid
    """
    errors = []

    # Check if problem exists
    if problem is None:
        return ["Problem is None"]

    # Check if start node exists
    if problem.start is None:
        errors.append("Start node is None")

    # Check if clients list exists and is not empty
    if not problem.clients:
        errors.append("No clients defined")
    else:
        # Check if start node is in clients list
        if problem.start not in problem.clients:
            errors.append("Start node not found in clients list")

        # Check client IDs
        client_ids = set(c.id for c in problem.clients)
        if len(client_ids) != len(problem.clients):
            errors.append("Duplicate client IDs found")

        # Check time windows
        for client in problem.clients:
            if client.earliness < 0:
                errors.append(f"Negative earliest time for client {client.id}")
            if client.tardiness < client.earliness:
                errors.append(f"Invalid time window for client {client.id}")
            if client.service_time < 0:
                errors.append(f"Negative service time for client {client.id}")

        # Check travel time matrix
        for i, client1 in enumerate(problem.clients):
            for j, client2 in enumerate(problem.clients):
                if i != j:
                    if client2.id not in client1.travel_times:
                        errors.append(
                            f"Missing travel time from {client1.id} to {client2.id}"
                        )
                    elif client1.travel_times[client2.id] < 0:
                        errors.append(
                            f"Negative travel time from {client1.id} to {client2.id}"
                        )

    # Check constraints
    if not problem.constraints:
        errors.append("No constraints defined")

    return errors


def validate_constraints(problem: "TSPTWProblem") -> List[str]:
    """Validate the constraints of a problem instance.

    Args:
        problem: The problem instance to validate

    Returns:
        List[str]: List of validation error messages, empty if constraints are valid
    """
    errors = []

    # Check if problem exists
    if problem is None:
        return ["Problem is None"]

    # Check if constraints exist
    if not problem.constraints:
        errors.append("No constraints defined")
        return errors

    # Validate each constraint
    for i, constraint in enumerate(problem.constraints):
        # Check if constraint is None
        if constraint is None:
            errors.append(f"Constraint {i} is None")
            continue

        # Check if constraint has required methods
        if not hasattr(constraint, "get_violation"):
            errors.append(f"Constraint {i} missing get_violation method")
        if not hasattr(constraint, "get_penalty"):
            errors.append(f"Constraint {i} missing get_penalty method")
        if not hasattr(constraint, "check"):
            errors.append(f"Constraint {i} missing check method")

    return errors


def is_valid_solution(solution: "PermuSolution", problem: "TSPTWProblem") -> bool:
    """Check if a solution is valid for a problem instance.

    Args:
        solution: The solution to check
        problem: The problem instance

    Returns:
        bool: True if solution is valid, False otherwise
    """
    return len(validate_solution(solution, problem)) == 0


def is_valid_problem(problem: "TSPTWProblem") -> bool:
    """Check if a problem instance is valid.

    Args:
        problem: The problem instance to check

    Returns:
        bool: True if problem is valid, False otherwise
    """
    return (
        len(validate_problem(problem)) == 0 and len(validate_constraints(problem)) == 0
    )
