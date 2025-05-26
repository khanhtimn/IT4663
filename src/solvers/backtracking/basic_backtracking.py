from typing import List, Optional, Set
import time
from ...core import TSPTWProblem, PermuSolution, Client
from ..base import Solver


class BacktrackingSolver(Solver):
    """Backtracking solver for TSPTW problem.

    This solver uses a backtracking algorithm to find the optimal solution
    for small TSPTW instances. It explores all possible permutations of clients
    while respecting time windows and other constraints.
    """

    def __init__(self, problem: TSPTWProblem, max_time: float = 300.0):
        """Initialize the backtracking solver.

        Args:
            problem: The TSPTW problem instance
            max_time: Maximum time limit in seconds (default: 300.0)
        """
        super().__init__(problem)
        self.max_time = max_time
        self.visited: Set[int] = set()
        self.current_route: List[Client] = []
        self.current_time = 0.0
        self.start_time = 0.0
        self.nodes_explored = 0
        self.max_depth = len(problem.clients) + 1  # +1 for depot
        self.last_progress_update = 0.0
        self.progress_interval = 1.0  # Update progress every second

        # Precompute minimum values for pruning
        self.min_service_time = min(c.service_time for c in problem.clients)
        self.min_travel_time = float("inf")
        for c1 in problem.clients:
            for c2 in problem.clients:
                if c1.id != c2.id and c2.id in c1.travel_times:
                    self.min_travel_time = min(
                        self.min_travel_time, c1.travel_times[c2.id]
                    )

        # Precompute earliest possible arrival times to each client
        self.earliest_arrival = {}
        for client in problem.clients:
            if client.id == problem.start.id:
                self.earliest_arrival[client.id] = 0
            else:
                min_travel = float("inf")
                for c in problem.clients:
                    if client.id in c.travel_times:
                        min_travel = min(min_travel, c.travel_times[client.id])
                self.earliest_arrival[client.id] = min_travel

        # Precompute reachable clients for each client
        self.reachable_clients = {}
        for c1 in problem.clients:
            self.reachable_clients[c1.id] = set()
            for c2 in problem.clients:
                if c1.id != c2.id and c2.id in c1.travel_times:
                    # Check if time window allows visiting c2 after c1
                    min_arrival = (
                        c1.earliness + c1.service_time + c1.travel_times[c2.id]
                    )
                    if min_arrival <= c2.tardiness:
                        self.reachable_clients[c1.id].add(c2.id)

        # Precompute minimum completion time for each subset of clients
        self.min_completion_times = {}
        self._precompute_min_completion_times()

    def _precompute_min_completion_times(self):
        """Precompute minimum completion times for all possible client subsets."""
        from itertools import combinations

        # For each possible subset size
        for size in range(1, len(self.problem.clients)):
            # For each subset of that size
            for subset in combinations(range(1, len(self.problem.clients)), size):
                min_time = float("inf")
                # Try each client as the last one in the subset
                for last_client_id in subset:
                    # Calculate minimum time to visit all clients in subset
                    # ending at last_client_id
                    time = self._calculate_min_subset_time(subset, last_client_id)
                    min_time = min(min_time, time)
                self.min_completion_times[subset] = min_time

    def _calculate_min_subset_time(self, subset: tuple, last_client_id: int) -> float:
        """Calculate minimum time to visit a subset of clients ending at last_client_id."""
        if not subset:
            return 0.0

        # Create a list of clients in the subset
        clients = [self.problem.clients[i] for i in subset]

        # Calculate minimum time to visit all clients
        min_time = 0.0
        current_pos = self.problem.start.id

        # Visit each client in the subset
        for client in clients:
            if client.id == last_client_id:
                continue

            # Find minimum travel time to this client
            min_travel = float("inf")
            for c in self.problem.clients:
                if client.id in c.travel_times:
                    min_travel = min(min_travel, c.travel_times[client.id])

            # Update minimum time
            min_time += min_travel + client.service_time

            # Update current position
            current_pos = client.id

        # Add time to reach last client
        if last_client_id in self.problem.clients[current_pos].travel_times:
            min_time += self.problem.clients[current_pos].travel_times[last_client_id]

        return min_time

    def solve(self) -> str:
        """Solve the TSPTW problem using backtracking.

        Returns:
            str: Status of the solution ("FEASIBLE" or "INFEASIBLE")
        """
        self.start_time = time.time()
        self.best_solution = None
        self.best_cost = float("inf")
        self.visited = {self.problem.start.id}
        self.current_route = [self.problem.start]
        self.current_time = 0.0
        self.nodes_explored = 0
        self.last_progress_update = self.start_time

        try:
            # Start backtracking from the depot
            self._backtrack(1)  # Start from step 1 (after depot)
        except TimeoutError:
            print(f"\nTimeout reached after {self.max_time} seconds")
            return self.Status.TIMEOUT
        except Exception as e:
            print(f"\nError during solving: {str(e)}")
            return self.Status.ERROR

        if self.best_solution is None:
            return self.Status.INFEASIBLE

        # Validate final solution
        if not self.problem.check(self.best_solution):
            print("Warning: Best solution found is not feasible")
            return self.Status.INFEASIBLE

        return self.Status.FEASIBLE

    def _backtrack(self, step: int):
        """Recursive backtracking function to explore all possible routes.

        Args:
            step: Current step in the backtracking process
        """
        # Check time limit and depth limit
        current_time = time.time()
        if current_time - self.start_time > self.max_time:
            raise TimeoutError("Maximum time limit exceeded")
        if step > self.max_depth:
            return

        # Update progress
        if current_time - self.last_progress_update >= self.progress_interval:
            self._update_progress(step)
            self.last_progress_update = current_time

        self.nodes_explored += 1
        last_client = self.current_route[-1]

        # Try each unvisited client that is reachable from the last client
        for client in self.problem.clients:
            if (
                client.id in self.visited
                or client.id not in self.reachable_clients[last_client.id]
            ):
                continue

            # Calculate arrival time
            travel_time = last_client.travel_times[client.id]
            arrival_time = self.current_time + travel_time

            # Check time window constraint
            if arrival_time > client.tardiness:
                continue

            # Calculate start time (considering waiting)
            start_time = max(arrival_time, client.earliness)
            if start_time + client.service_time > client.tardiness:
                continue

            # Try this client
            old_time = self.current_time
            self.current_time = start_time + client.service_time
            self.visited.add(client.id)
            self.current_route.append(client)

            # If we've visited all clients, try to return to depot
            if len(self.visited) == len(self.problem.clients):
                if self.problem.start.id in client.travel_times:
                    # Create complete route with depot at start and end
                    complete_route = (
                        [self.problem.start]
                        + self.current_route[1:]
                        + [self.problem.start]
                    )

                    # Create solution with proper size
                    solution = PermuSolution(len(complete_route))
                    solution.route = complete_route

                    # Check if solution is feasible
                    if self.problem.check(solution):
                        total_cost = self.problem.cal_cost(solution)
                        if total_cost < self.best_cost:
                            self.best_solution = solution
                            self.best_cost = total_cost
                            self._update_progress(step, True)
            else:
                # Enhanced pruning strategies
                if self._can_lead_to_better_solution(client):
                    self._backtrack(step + 1)

            # Backtrack
            self.current_time = old_time
            self.visited.remove(client.id)
            self.current_route.pop()

    def _can_lead_to_better_solution(self, last_client: Client) -> bool:
        """Check if current partial solution can lead to a better solution.

        Args:
            last_client: The last client added to the route

        Returns:
            bool: True if the current partial solution can lead to a better solution
        """
        # Basic pruning: Check if current time plus minimum remaining time exceeds best cost
        remaining_clients = len(self.problem.clients) - len(self.visited)
        min_remaining_time = remaining_clients * (
            self.min_service_time + self.min_travel_time
        )

        # Add minimum time to return to depot
        if self.problem.start.id in last_client.travel_times:
            min_remaining_time += last_client.travel_times[self.problem.start.id]
        else:
            return False  # Can't return to depot

        if self.current_time + min_remaining_time >= self.best_cost:
            return False

        # Check if remaining clients can be visited within their time windows
        for client in self.problem.clients:
            if client.id not in self.visited:
                # Calculate earliest possible arrival time
                min_arrival = self.current_time + self.earliest_arrival[client.id]
                if min_arrival > client.tardiness:
                    return False

                # Check if this client is reachable from any remaining client
                is_reachable = False
                for c in self.problem.clients:
                    if (
                        c.id not in self.visited
                        and client.id in self.reachable_clients[c.id]
                    ):
                        is_reachable = True
                        break
                if not is_reachable:
                    return False

        # Check minimum completion time for remaining clients
        remaining_ids = tuple(
            c.id for c in self.problem.clients if c.id not in self.visited
        )
        if remaining_ids in self.min_completion_times:
            min_completion = self.min_completion_times[remaining_ids]
            if self.current_time + min_completion >= self.best_cost:
                return False

        return True

    def _update_progress(self, step: int, found_better: bool = False):
        """Update and display progress information.

        Args:
            step: Current step in the backtracking process
            found_better: Whether a better solution was just found
        """
        elapsed = time.time() - self.start_time
        progress = (step / self.max_depth) * 100
        print(
            f"\rProgress: {progress:.1f}% | "
            f"Nodes explored: {self.nodes_explored} | "
            f"Time: {elapsed:.1f}s | "
            f"Best cost: {self.best_cost if self.best_solution else 'inf'}"
            + (" *" if found_better else ""),
            end="",
        )

    def get_best_solution(self) -> Optional[PermuSolution]:
        """Get the best solution found.

        Returns:
            Optional[PermuSolution]: The best solution found, or None if no solution was found
        """
        return self.best_solution

    def get_stats(self) -> dict:
        """Get solver statistics.

        Returns:
            dict: Dictionary containing solver statistics
        """
        return {
            "best_cost": self.best_cost if self.best_solution else float("inf"),
            "nodes_explored": self.nodes_explored,
            "time_elapsed": time.time() - self.start_time
            if hasattr(self, "start_time")
            else 0.0,
            "max_depth": self.max_depth,
            "min_service_time": self.min_service_time,
            "min_travel_time": self.min_travel_time,
        }
