import random
import time
from typing import List, Tuple
from ...core.problem import TSPTWProblem
from ...core.solution import PermuSolution
from ...core.client import Client
from ..base import Solver


class GASolver(Solver):
    """Genetic Algorithm solver for TSPTW."""

    def __init__(
        self,
        problem: TSPTWProblem,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        elite_size: int = 5,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
    ):
        super().__init__(problem)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.best_solution = None
        self.solve_time = 0.0
        self.start_time = 0.0
        self.generation_best_costs = []
        self.generation_avg_costs = []
        self.generation_feasible_ratio = []

    def solve(self, **kwargs) -> Solver.Status:
        """Solve the TSPTW problem using genetic algorithm.

        Args:
            **kwargs: Additional solver parameters.

        Returns:
            Solver.Status: The status of the solution attempt.
        """
        self.start_time = time.time()

        # Initialize population
        population = self._initialize_population()

        # Initialize best solution with the best from initial population
        fitness_scores = self._evaluate_population(population)
        best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        self.best_solution = population[best_idx]

        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_population(population)

            # Track statistics
            self._track_generation_stats(population)

            # Create new generation
            new_population = []

            # Elitism - keep best solutions
            sorted_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True,
            )
            new_population.extend(
                [population[i] for i in sorted_indices[: self.elite_size]]
            )

            # Create offspring through selection, crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    # If no crossover, copy better parent
                    child = (
                        parent1
                        if fitness_scores[population.index(parent1)]
                        > fitness_scores[population.index(parent2)]
                        else parent2
                    )

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

            # Update best solution
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            current_best = population[best_idx]
            if current_best.cost < self.best_solution.cost:
                self.best_solution = current_best

            # Print progress
            if generation % 10 == 0:
                self._print_progress(generation)

        self.solve_time = time.time() - self.start_time
        return (
            Solver.Status.FEASIBLE
            if self.best_solution.is_feasible()
            else Solver.Status.INFEASIBLE
        )

    def _initialize_population(self) -> List[PermuSolution]:
        """Initialize population with random solutions."""
        population = []
        for _ in range(self.population_size):
            route = [self.problem.start]
            clients = self.problem.clients[1:]  # Exclude start
            random.shuffle(clients)
            route.extend(clients)
            route.append(self.problem.start)

            # Create solution
            solution = PermuSolution(len(route))
            solution.route = route
            solution.cost = self.problem.cal_cost(solution)
            solution.violations = self.problem.cal_violations(solution)
            solution.penalty = self.problem.cal_penalty(solution)
            population.append(solution)
        return population

    def _evaluate_population(self, population: List[PermuSolution]) -> List[float]:
        """Evaluate fitness of each solution in population."""
        fitness_scores = []
        for solution in population:
            if solution.is_feasible():
                # Higher fitness for lower cost
                fitness = 1.0 / (1.0 + solution.cost)
            else:
                # Penalize infeasible solutions based on violations and penalty
                # Use a smaller denominator to ensure infeasible solutions have lower fitness
                fitness = 1.0 / (1.0 + solution.penalty * 10.0)
            fitness_scores.append(fitness)
        return fitness_scores

    def _tournament_select(
        self,
        population: List[PermuSolution],
        fitness_scores: List[float],
    ) -> PermuSolution:
        """Select a single parent using tournament selection."""
        # Select tournament participants
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament = [(population[i], fitness_scores[i]) for i in tournament_indices]
        # Select winner (highest fitness)
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(
        self, parent1: PermuSolution, parent2: PermuSolution
    ) -> PermuSolution:
        """Perform order crossover between two parent solutions."""
        route1 = parent1.route[1:-1]  # Exclude start/end
        route2 = parent2.route[1:-1]

        # Select random segment
        start = random.randint(0, len(route1) - 1)
        length = random.randint(1, len(route1) - start)
        segment = route1[start : start + length]

        # Create child route
        child_route = []
        remaining = [c for c in route2 if c not in segment]

        # Insert segment at same position as parent1
        child_route.extend(remaining[:start])
        child_route.extend(segment)
        child_route.extend(remaining[start:])

        # Add back start/end
        child_route = [self.problem.start] + child_route + [self.problem.start]

        # Create solution
        solution = PermuSolution(len(child_route))
        solution.route = child_route
        solution.cost = self.problem.cal_cost(solution)
        solution.violations = self.problem.cal_violations(solution)
        solution.penalty = self.problem.cal_penalty(solution)
        return solution

    def _mutate(self, solution: PermuSolution) -> PermuSolution:
        """Perform mutation on solution using one of several operators."""
        route = solution.route[1:-1]  # Exclude start/end
        if len(route) < 2:
            return solution

        # Randomly select mutation operator
        mutation_type = random.random()

        if mutation_type < 0.4:  # 40% chance for swap mutation
            # Select random positions to swap
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        elif mutation_type < 0.7:  # 30% chance for reverse mutation
            # Select random segment to reverse
            start = random.randint(0, len(route) - 2)
            end = random.randint(start + 1, len(route))
            route[start:end] = reversed(route[start:end])
        else:  # 30% chance for shift mutation
            # Select random segment to shift
            start = random.randint(0, len(route) - 2)
            length = random.randint(1, min(3, len(route) - start))
            segment = route[start : start + length]
            del route[start : start + length]
            insert_pos = random.randint(0, len(route))
            route[insert_pos:insert_pos] = segment

        # Add back start/end
        route = [solution.route[0]] + route + [solution.route[-1]]

        # Create solution
        new_solution = PermuSolution(len(route))
        new_solution.route = route
        new_solution.cost = self.problem.cal_cost(new_solution)
        new_solution.violations = self.problem.cal_violations(new_solution)
        new_solution.penalty = self.problem.cal_penalty(new_solution)
        return new_solution

    def _track_generation_stats(self, population: List[PermuSolution]):
        """Track statistics for the current generation."""
        costs = [s.cost for s in population]
        self.generation_best_costs.append(min(costs))
        self.generation_avg_costs.append(sum(costs) / len(costs))
        feasible_count = sum(1 for s in population if s.is_feasible())
        self.generation_feasible_ratio.append(feasible_count / len(population))

    def _print_progress(self, generation: int):
        """Print progress information."""
        elapsed = time.time() - self.start_time
        progress = (generation / self.generations) * 100
        best_cost = self.best_solution.cost if self.best_solution else float("inf")
        print(
            f"\rGeneration {generation}/{self.generations} | "
            f"Progress: {progress:.1f}% | "
            f"Time: {elapsed:.1f}s | "
            f"Best cost: {best_cost:.2f} | "
            f"Feasible: {self.generation_feasible_ratio[-1] * 100:.1f}%",
            end="",
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
            "iterations": self.generations,  # Add iterations for benchmark compatibility
            "generations": self.generations,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_size": self.elite_size,
            "tournament_size": self.tournament_size,
            "generation_best_costs": self.generation_best_costs,
            "generation_avg_costs": self.generation_avg_costs,
            "generation_feasible_ratio": self.generation_feasible_ratio,
            "best_solution_feasible": self.best_solution.is_feasible()
            if self.best_solution
            else False,
            "final_population_size": self.population_size,
            "final_feasible_ratio": self.generation_feasible_ratio[-1]
            if self.generation_feasible_ratio
            else 0.0,
        }
