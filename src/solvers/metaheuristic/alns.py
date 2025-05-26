import random
import time
import math
from typing import List, Tuple, Callable
from ...core.problem import TSPTWProblem
from ...core.solution import PermuSolution
from ...core.client import Client
from ..base import Solver
from ..utils.solutions import create_initial_solution, accept_solution_sa


class ALNSSolver(Solver):
    """Adaptive Large Neighborhood Search solver for TSPTW."""

    def __init__(
        self,
        problem: TSPTWProblem,
        iterations: int = 1000,
        segment_size: int = 3,
        temperature: float = 100.0,
        cooling_rate: float = 0.99,
        weight_update_interval: int = 100,
    ):
        super().__init__(problem)
        self.iterations = iterations
        self.segment_size = segment_size
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.weight_update_interval = weight_update_interval
        self.best_solution = None
        self.solve_time = 0.0
        self.start_time = 0.0

        # Initialize operator weights
        self.destroy_weights = [1.0] * 3  # random, worst, related
        self.repair_weights = [1.0] * 3  # greedy, regret, random

        # Initialize operator scores
        self.destroy_scores = [0.0] * 3
        self.repair_scores = [0.0] * 3

        # Track statistics
        self.iteration_best_costs = []
        self.iteration_avg_costs = []
        self.iteration_feasible_ratio = []

    def solve(self, **kwargs) -> Solver.Status:
        """Solve the TSPTW problem using ALNS.

        Args:
            **kwargs: Additional solver parameters.

        Returns:
            Solver.Status: The status of the solution attempt.
        """
        self.start_time = time.time()

        # Initialize solution using existing utility function
        current_solution = create_initial_solution(self.problem)
        self.best_solution = current_solution

        # Main ALNS loop
        for iteration in range(self.iterations):
            # Select destroy and repair operators
            destroy_idx = self._select_operator_index(self.destroy_weights)
            repair_idx = self._select_operator_index(self.repair_weights)

            # Create new solution
            new_solution = self._create_new_solution(
                current_solution, destroy_idx, repair_idx
            )

            # Accept or reject new solution using existing utility function
            if accept_solution_sa(current_solution, new_solution, self.temperature):
                current_solution = new_solution
                if current_solution.cost < self.best_solution.cost:
                    self.best_solution = current_solution
                    # Update operator scores
                    self.destroy_scores[destroy_idx] += 1.0
                    self.repair_scores[repair_idx] += 1.0

            # Update temperature
            self.temperature *= self.cooling_rate

            # Update operator weights periodically
            if (iteration + 1) % self.weight_update_interval == 0:
                self._update_operator_weights()

            # Track statistics
            self._track_iteration_stats(current_solution)

            # Print progress
            if iteration % 10 == 0:
                self._print_progress(iteration)

        self.solve_time = time.time() - self.start_time
        return (
            Solver.Status.FEASIBLE
            if self.best_solution.is_feasible()
            else Solver.Status.INFEASIBLE
        )

    def _select_operator_index(self, weights: List[float]) -> int:
        """Select operator index using roulette wheel selection."""
        total_weight = sum(weights)
        if total_weight == 0:
            return random.randint(0, len(weights) - 1)

        r = random.random() * total_weight
        cumsum = 0

        for i, weight in enumerate(weights):
            cumsum += weight
            if cumsum >= r:
                return i

        return len(weights) - 1

    def _update_operator_weights(self):
        """Update operator weights based on their performance."""
        # Update destroy operator weights
        for i in range(len(self.destroy_weights)):
            if self.destroy_scores[i] > 0:
                self.destroy_weights[i] = (
                    0.8 * self.destroy_weights[i] + 0.2 * self.destroy_scores[i]
                )
            else:
                self.destroy_weights[i] *= 0.8
            self.destroy_scores[i] = 0

        # Update repair operator weights
        for i in range(len(self.repair_weights)):
            if self.repair_scores[i] > 0:
                self.repair_weights[i] = (
                    0.8 * self.repair_weights[i] + 0.2 * self.repair_scores[i]
                )
            else:
                self.repair_weights[i] *= 0.8
            self.repair_scores[i] = 0

    def _print_progress(self, iteration: int):
        """Print progress information."""
        elapsed = time.time() - self.start_time
        progress = (iteration / self.iterations) * 100
        best_cost = self.best_solution.cost if self.best_solution else float("inf")
        print(
            f"\rIteration {iteration}/{self.iterations} | "
            f"Progress: {progress:.1f}% | "
            f"Time: {elapsed:.1f}s | "
            f"Best cost: {best_cost:.2f} | "
            f"Feasible: {self.iteration_feasible_ratio[-1] * 100:.1f}%",
            end="",
        )

    def _random_remove(
        self, solution: PermuSolution
    ) -> Tuple[PermuSolution, List[Client]]:
        """Remove random segment of clients from solution."""
        route = solution.route[1:-1]  # Exclude start/end
        if len(route) <= self.segment_size:
            return solution, []

        # Select random segment
        start_idx = random.randint(0, len(route) - self.segment_size)
        removed = route[start_idx : start_idx + self.segment_size]
        new_route = route[:start_idx] + route[start_idx + self.segment_size :]

        # Add back start/end
        new_route = [self.problem.start] + new_route + [self.problem.start]

        # Create new solution
        new_solution = PermuSolution(len(new_route))
        new_solution.route = new_route
        new_solution.cost = self.problem.cal_cost(new_solution)
        new_solution.violations = self.problem.cal_violations(new_solution)
        new_solution.penalty = self.problem.cal_penalty(new_solution)
        return new_solution, removed

    def _worst_remove(
        self, solution: PermuSolution
    ) -> Tuple[PermuSolution, List[Client]]:
        """Remove segment of clients with highest cost contribution."""
        route = solution.route[1:-1]  # Exclude start/end
        if len(route) <= self.segment_size:
            return solution, []

        # Calculate cost contribution of each client
        costs = []
        for i in range(len(route)):
            prev = solution.route[i]
            curr = route[i]
            next_ = solution.route[i + 2]
            cost = prev.travel_times.get(curr.id, float("inf")) + curr.travel_times.get(
                next_.id, float("inf")
            )
            costs.append((i, cost))

        # Remove clients with highest cost
        costs.sort(key=lambda x: x[1], reverse=True)
        removed_indices = sorted([i for i, _ in costs[: self.segment_size]])
        removed = [route[i] for i in removed_indices]

        # Create new route
        new_route = [route[i] for i in range(len(route)) if i not in removed_indices]
        new_route = [self.problem.start] + new_route + [self.problem.start]

        # Create new solution
        new_solution = PermuSolution(len(new_route))
        new_solution.route = new_route
        new_solution.cost = self.problem.cal_cost(new_solution)
        new_solution.violations = self.problem.cal_violations(new_solution)
        new_solution.penalty = self.problem.cal_penalty(new_solution)
        return new_solution, removed

    def _related_remove(
        self, solution: PermuSolution
    ) -> Tuple[PermuSolution, List[Client]]:
        """Remove a set of related customers from the solution."""
        route = solution.route[1:-1]  # Exclude start/end
        if len(route) < 2:
            return solution, []

        # Select a random customer
        seed = random.choice(route)
        removed = [seed]
        route.remove(seed)

        # Remove related customers
        while len(removed) < self.segment_size and route:
            # Find customer closest to any removed customer
            min_dist = float("inf")
            next_customer = None
            for customer in route:
                for removed_customer in removed:
                    dist = removed_customer.travel_times.get(customer.id, float("inf"))
                    if dist < min_dist:
                        min_dist = dist
                        next_customer = customer

            if next_customer is None:
                break

            removed.append(next_customer)
            route.remove(next_customer)

        # Create new solution
        new_route = [self.problem.start] + route + [self.problem.start]
        new_solution = PermuSolution(len(new_route))
        new_solution.route = new_route
        new_solution.cost = self.problem.cal_cost(new_solution)
        new_solution.violations = self.problem.cal_violations(new_solution)
        new_solution.penalty = self.problem.cal_penalty(new_solution)
        return new_solution, removed

    def _greedy_insert(self, removed: List[Client]) -> PermuSolution:
        """Insert clients greedily at best position."""
        if not removed:
            route = [self.problem.start, self.problem.start]
            solution = PermuSolution(len(route))
            solution.route = route
            solution.cost = self.problem.cal_cost(solution)
            solution.violations = self.problem.cal_violations(solution)
            solution.penalty = self.problem.cal_penalty(solution)
            return solution

        # Initialize route with start node
        route = [self.problem.start]

        # Insert each client
        for client in removed:
            best_cost = float("inf")
            best_pos = 0

            # Try inserting at each position
            for i in range(1, len(route)):
                new_route = route[:i] + [client] + route[i:]
                # Ensure route has at least start and one client
                if len(new_route) >= 2:
                    temp_solution = PermuSolution(len(new_route))
                    temp_solution.route = new_route
                    temp_solution.cost = self.problem.cal_cost(temp_solution)
                    temp_solution.violations = self.problem.cal_violations(
                        temp_solution
                    )
                    temp_solution.penalty = self.problem.cal_penalty(temp_solution)
                    if temp_solution.is_feasible() and temp_solution.cost < best_cost:
                        best_cost = temp_solution.cost
                        best_pos = i

            # Insert the client at the best position
            route.insert(best_pos, client)

        # Ensure route has start and end nodes
        if route[0] != self.problem.start:
            route.insert(0, self.problem.start)
        if route[-1] != self.problem.start:
            route.append(self.problem.start)

        # Create final solution
        solution = PermuSolution(len(route))
        solution.route = route
        solution.cost = self.problem.cal_cost(solution)
        solution.violations = self.problem.cal_violations(solution)
        solution.penalty = self.problem.cal_penalty(solution)
        return solution

    def _regret_insert(self, removed: List[Client]) -> PermuSolution:
        """Insert clients using regret heuristic."""
        if not removed:
            route = [self.problem.start, self.problem.start]
            solution = PermuSolution(len(route))
            solution.route = route
            solution.cost = self.problem.cal_cost(solution)
            solution.violations = self.problem.cal_violations(solution)
            solution.penalty = self.problem.cal_penalty(solution)
            return solution

        # Initialize route with start node
        route = [self.problem.start]

        # Insert each client
        for client in removed:
            # Calculate costs for all possible positions
            costs = []
            for i in range(1, len(route)):
                new_route = route[:i] + [client] + route[i:]
                if len(new_route) >= 2:
                    temp_solution = PermuSolution(len(new_route))
                    temp_solution.route = new_route
                    temp_solution.cost = self.problem.cal_cost(temp_solution)
                    temp_solution.violations = self.problem.cal_violations(
                        temp_solution
                    )
                    temp_solution.penalty = self.problem.cal_penalty(temp_solution)
                    if temp_solution.is_feasible():
                        costs.append((i, temp_solution.cost))

            if not costs:
                # If no feasible position found, insert at random position
                pos = random.randint(1, len(route))
                route.insert(pos, client)
            else:
                # Sort costs and calculate regret
                costs.sort(key=lambda x: x[1])
                if len(costs) >= 2:
                    regret = costs[1][1] - costs[0][1]
                    # Use regret to bias position selection
                    if random.random() < 0.8:  # 80% chance to choose best position
                        pos = costs[0][0]
                    else:
                        pos = random.choice([c[0] for c in costs])
                else:
                    pos = costs[0][0]
                route.insert(pos, client)

        # Ensure route has start and end nodes
        if route[0] != self.problem.start:
            route.insert(0, self.problem.start)
        if route[-1] != self.problem.start:
            route.append(self.problem.start)

        # Create final solution
        solution = PermuSolution(len(route))
        solution.route = route
        solution.cost = self.problem.cal_cost(solution)
        solution.violations = self.problem.cal_violations(solution)
        solution.penalty = self.problem.cal_penalty(solution)
        return solution

    def _random_insert(self, removed: List[Client]) -> PermuSolution:
        """Insert clients at random positions."""
        if not removed:
            route = [self.problem.start, self.problem.start]
            solution = PermuSolution(len(route))
            solution.route = route
            solution.cost = self.problem.cal_cost(solution)
            solution.violations = self.problem.cal_violations(solution)
            solution.penalty = self.problem.cal_penalty(solution)
            return solution

        # Initialize route with start node
        route = [self.problem.start]

        # Insert each client at a random position
        for client in removed:
            # Try to find a feasible position
            feasible_positions = []
            for i in range(1, len(route)):
                new_route = route[:i] + [client] + route[i:]
                if len(new_route) >= 2:
                    temp_solution = PermuSolution(len(new_route))
                    temp_solution.route = new_route
                    temp_solution.cost = self.problem.cal_cost(temp_solution)
                    temp_solution.violations = self.problem.cal_violations(
                        temp_solution
                    )
                    temp_solution.penalty = self.problem.cal_penalty(temp_solution)
                    if temp_solution.is_feasible():
                        feasible_positions.append(i)

            # Insert at random feasible position or random position if no feasible ones
            if feasible_positions:
                pos = random.choice(feasible_positions)
            else:
                pos = random.randint(1, len(route))
            route.insert(pos, client)

        # Ensure route has start and end nodes
        if route[0] != self.problem.start:
            route.insert(0, self.problem.start)
        if route[-1] != self.problem.start:
            route.append(self.problem.start)

        # Create final solution
        solution = PermuSolution(len(route))
        solution.route = route
        solution.cost = self.problem.cal_cost(solution)
        solution.violations = self.problem.cal_violations(solution)
        solution.penalty = self.problem.cal_penalty(solution)
        return solution

    def _create_new_solution(
        self, current_solution: PermuSolution, destroy_idx: int, repair_idx: int
    ) -> PermuSolution:
        """Create a new solution by applying destroy and repair operators."""
        # Apply destroy operator
        if destroy_idx == 0:
            partial_solution, removed = self._random_remove(current_solution)
        elif destroy_idx == 1:
            partial_solution, removed = self._worst_remove(current_solution)
        else:
            partial_solution, removed = self._related_remove(current_solution)

        # Apply repair operator
        if repair_idx == 0:
            new_solution = self._greedy_insert(removed)
        elif repair_idx == 1:
            new_solution = self._regret_insert(removed)
        else:
            new_solution = self._random_insert(removed)

        return new_solution

    def _track_iteration_stats(self, current_solution: PermuSolution):
        """Track statistics for the current iteration."""
        self.iteration_best_costs.append(self.best_solution.cost)
        self.iteration_avg_costs.append(current_solution.cost)
        self.iteration_feasible_ratio.append(
            1.0 if current_solution.is_feasible() else 0.0
        )

    def get_stats(self) -> dict:
        """Get solver statistics.

        Returns:
            dict: Dictionary containing solver statistics
        """
        return {
            "best_cost": self.best_solution.cost
            if self.best_solution
            else float("inf"),
            "solve_time": self.solve_time,
            "iterations": self.iterations,
            "segment_size": self.segment_size,
            "initial_temperature": self.temperature,
            "cooling_rate": self.cooling_rate,
            "weight_update_interval": self.weight_update_interval,
            "destroy_weights": self.destroy_weights,
            "repair_weights": self.repair_weights,
            "iteration_best_costs": self.iteration_best_costs,
            "iteration_avg_costs": self.iteration_avg_costs,
            "iteration_feasible_ratio": self.iteration_feasible_ratio,
            "best_solution_feasible": self.best_solution.is_feasible()
            if self.best_solution
            else False,
            "final_temperature": self.temperature,
        }
