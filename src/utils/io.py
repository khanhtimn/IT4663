from typing import List, Tuple
from ..core.problem import TSPTWProblem
from ..core.solution import PermuSolution
from ..core.client import Client


def read_input_file(filepath: str) -> TSPTWProblem:
    """Read TSPTW problem instance from file.

    Args:
        filepath: Path to input file.

    Returns:
        TSPTWProblem: The problem instance.
    """
    with open(filepath, "r") as f:
        # Read number of customers
        n = int(f.readline().strip())

        # Read customer data
        clients: List[Client] = []
        depot = Client(id=0, time_window=(0, float("inf"), 0))
        clients.append(depot)

        for i in range(1, n + 1):
            e, l, d = map(int, f.readline().strip().split())
            clients.append(Client(id=i, time_window=(e, l, d)))

        # Read travel time matrix
        for i in range(n + 1):
            row = list(map(int, f.readline().strip().split()))
            for j in range(n + 1):
                if i != j:
                    clients[i].add_travel_time(clients[j].id, row[j])

        return TSPTWProblem(clients=clients, start_at=depot)


def read_console() -> TSPTWProblem:
    """Read TSPTW problem instance from console input.

    Returns:
        TSPTWProblem: The problem instance.

    Raises:
        ValueError: If input format is invalid.
    """
    try:
        # Read number of customers
        n = int(input().strip())
        if n <= 0:
            raise ValueError("Number of customers must be positive")

        # Read customer data
        clients: List[Client] = []
        depot = Client(id=0, time_window=(0, float("inf"), 0))
        clients.append(depot)

        for i in range(1, n + 1):
            try:
                e, l, d = map(int, input().strip().split())
                if e < 0 or l <= e or d < 0:
                    raise ValueError(
                        f"Invalid time window or service time for customer {i}"
                    )
                clients.append(Client(id=i, time_window=(e, l, d)))
            except ValueError as e:
                raise ValueError(f"Invalid input for customer {i}: {str(e)}")

        # Read travel time matrix
        for i in range(n + 1):
            try:
                row = list(map(int, input().strip().split()))
                if len(row) != n + 1:
                    raise ValueError(f"Invalid number of travel times for location {i}")
                for j in range(n + 1):
                    if i != j:
                        if row[j] < 0:
                            raise ValueError(
                                f"Negative travel time between locations {i} and {j}"
                            )
                        clients[i].add_travel_time(clients[j].id, row[j])
            except ValueError as e:
                raise ValueError(f"Invalid travel time matrix row {i}: {str(e)}")

        return TSPTWProblem(clients=clients, start_at=depot)
    except ValueError as e:
        raise ValueError(f"Error reading input: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading input: {str(e)}")


def write_solution(solution: PermuSolution, filepath: str = None):
    """Write solution to file or console.

    Args:
        solution: The solution to write.
        filepath: Optional path to output file. If None, writes to console.
    """
    route = solution.route
    n = len(route) - 2  # Exclude depot at start and end

    # Format solution details
    details = [
        f"Number of customers: {n}",
        f"Route: {' '.join(str(c.id) for c in route[1:-1])}",
        f"Total cost: {solution.cost:.2f}",
        f"Constraint violations: {solution.violations}",
        f"Total penalty: {solution.penalty:.2f}",
        f"Feasible: {solution.is_feasible()}",
    ]

    if filepath:
        with open(filepath, "w") as f:
            f.write("\n".join(details) + "\n")
    else:
        print("\n".join(details))
