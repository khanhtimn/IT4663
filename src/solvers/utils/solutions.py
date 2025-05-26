"""Solution manipulation utilities for TSPTW solvers."""

import random
import math
from typing import List, Tuple
from ...core.problem import TSPTWProblem
from ...core.solution import PermuSolution
from ...core.client import Client


def create_initial_solution(problem: TSPTWProblem) -> PermuSolution:
    """Create initial solution using nearest neighbor heuristic.

    Args:
        problem: The TSPTW problem instance

    Returns:
        PermuSolution: Initial solution using nearest neighbor heuristic
    """
    unvisited = set(problem.clients)
    route = [problem.start]
    current = problem.start
    unvisited.remove(problem.start)  # Remove start from unvisited

    while unvisited:
        # Find nearest unvisited client
        nearest = min(
            unvisited, key=lambda c: current.travel_times.get(c.id, float("inf"))
        )
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    route.append(problem.start)  # Return to start

    # Create solution
    solution = PermuSolution(len(route))
    solution.route = route
    solution.cost = problem.cal_cost(solution)
    solution.violations = problem.cal_violations(solution)
    solution.penalty = problem.cal_penalty(solution)
    return solution


def apply_2opt_move(
    problem: TSPTWProblem, solution: PermuSolution, i: int, j: int
) -> PermuSolution:
    """Apply 2-opt move to solution.

    Args:
        problem: The TSPTW problem instance
        solution: Current solution
        i: First position in route
        j: Second position in route

    Returns:
        PermuSolution: New solution after 2-opt move
    """
    route = solution.route
    new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]

    # Create new solution
    new_solution = PermuSolution(len(new_route))
    new_solution.route = new_route
    new_solution.cost = problem.cal_cost(new_solution)
    new_solution.violations = problem.cal_violations(new_solution)
    new_solution.penalty = problem.cal_penalty(new_solution)
    return new_solution


def apply_swap_move(
    problem: TSPTWProblem, solution: PermuSolution, i: int, j: int
) -> PermuSolution:
    """Apply swap move to solution.

    Args:
        problem: The TSPTW problem instance
        solution: Current solution
        i: First position in route
        j: Second position in route

    Returns:
        PermuSolution: New solution after swap move
    """
    route = solution.route.copy()
    route[i], route[j] = route[j], route[i]

    # Create new solution
    new_solution = PermuSolution(len(route))
    new_solution.route = route
    new_solution.cost = problem.cal_cost(new_solution)
    new_solution.violations = problem.cal_violations(new_solution)
    new_solution.penalty = problem.cal_penalty(new_solution)
    return new_solution


def accept_solution_sa(
    current: PermuSolution, new: PermuSolution, temperature: float
) -> bool:
    """Accept solution based on simulated annealing criterion.

    Args:
        current: Current solution
        new: New solution to evaluate
        temperature: Current temperature

    Returns:
        bool: True if solution should be accepted
    """
    if new.is_feasible() and not current.is_feasible():
        return True

    if not new.is_feasible() and current.is_feasible():
        return False

    if new.is_feasible():
        delta = new.cost - current.cost
    else:
        delta = new.penalty - current.penalty

    if delta <= 0:
        return True

    return random.random() < math.exp(-delta / temperature)
