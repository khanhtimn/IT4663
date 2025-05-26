from typing import Dict, Tuple


class Client:
    """Represents a client in the TSPTW problem.

    Each client has an ID, time window constraints (earliness, tardiness),
    service time, and travel times to other clients.
    """

    def __init__(self, id: int | str, time_window: Tuple[int, int, int]):
        """Initialize a new client.

        Args:
            id: Unique identifier for the client
            time_window: Tuple of (earliness, tardiness, service_time)
                - earliness: Earliest time the client can be visited
                - tardiness: Latest time the client can be visited
                - service_time: Time required to service this client
        """
        self.id = int(id)
        self.earliness = time_window[0]
        self.tardiness = time_window[1]
        self.service_time = time_window[2]
        self.travel_times: Dict[int | str, int] = {}

    def add_travel_time(self, other_id: int | str, travel_time: int):
        """Add travel time to another client.

        Args:
            other_id: ID of the other client
            travel_time: Time required to travel to the other client
        """
        self.travel_times[other_id] = travel_time

    def __eq__(self, value):
        """Check if this client is equal to another client.

        Args:
            value: Other client to compare with

        Returns:
            bool: True if clients have the same ID
        """
        if not isinstance(value, Client):
            return False
        return self.id == value.id

    def __hash__(self):
        """Get hash value for this client.

        Returns:
            int: Hash value based on client ID
        """
        return hash(self.id)

    def __str__(self):
        """Get string representation of this client.

        Returns:
            str: String showing client details and travel times
        """
        travel_str = ", ".join(f"{c}: {t}" for c, t in self.travel_times.items())
        return f"Client({self.id}, e={self.earliness}, t={self.tardiness}, d={self.service_time}, times=[{travel_str}])"
